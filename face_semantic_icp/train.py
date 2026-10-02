from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .geometry import Mesh, load_mesh, write_obj
from .metrics import chamfer_distance, normal_consistency
from .pairs import PairRecord, read_pairs_manifest
from .template import load_template
from .viewer import write_viewer


@dataclass
class TrainConfig:
    pairs: str
    out: str = "outputs/train_deformnet"
    template: str | None = None
    epochs: int = 5
    batch_size: int = 2
    lr: float = 1e-3
    device: str = "cuda"
    latent_dim: int = 96
    hidden_dim: int = 128
    max_pairs: int | None = None
    l1_weight: float = 1.0
    l2_weight: float = 0.15
    chamfer_weight: float = 0.05
    edge_weight: float = 0.03
    laplacian_weight: float = 0.05
    landmark_weight: float = 0.5
    model_version: int = 2


def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train/eval. Install torch or use the non-learning pipeline commands.") from exc
    return torch, nn, DataLoader, Dataset


def _device(name: str):
    torch, _, _, _ = _torch()
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


class DeformNetDataset:
    def __init__(self, records: list[PairRecord], target_vertex_count: int):
        self.records = records
        self.target_vertex_count = target_vertex_count

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        points = np.load(record.source_points)["points"].astype(np.float32)
        target = np.load(record.target_vertices).astype(np.float32)
        if target.shape[0] != self.target_vertex_count:
            raise ValueError(f"Pair {record.pair_id} target has {target.shape[0]} vertices, expected {self.target_vertex_count}")
        return points, target, index


def build_gcn_adj(faces: np.ndarray, vertex_count: int):
    """Build D^{-1/2}(A+I)D^{-1/2} normalised adjacency as a sparse CPU tensor.

    No PyTorch Geometric needed — pure scipy + torch.sparse.
    """
    import scipy.sparse as sp
    torch, _, _, _ = _torch()

    rows: list[int] = []
    cols: list[int] = []
    for f in faces:
        for a, b in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
            rows += [a, b]
            cols += [b, a]
    rows += list(range(vertex_count))
    cols += list(range(vertex_count))
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count)).tocsr()
    A.data = np.ones_like(A.data)
    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5, where=deg > 0), 0.0)
    D = sp.diags(d_inv_sqrt)
    A_norm = (D @ A @ D).tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    val = torch.tensor(A_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (vertex_count, vertex_count)).coalesce()


def build_model(vertex_count: int, latent_dim: int, hidden_dim: int, version: int = 2, template_faces: np.ndarray | None = None):
    torch, nn, _, _ = _torch()
    import torch.nn.functional as F

    if version == 1:
        class PointDeformNetV1(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(3, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim + 3, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )

            def forward(self, points, template_vertices):
                latent = self.encoder(points).max(dim=1).values
                lat_exp = latent[:, None, :].expand(-1, template_vertices.shape[0], -1)
                tpl_exp = template_vertices[None, :, :].expand(points.shape[0], -1, -1)
                return tpl_exp + self.decoder(torch.cat([tpl_exp, lat_exp], dim=-1))

            def predict_with_uncertainty(self, points, template_vertices):
                pred = self.forward(points, template_vertices)
                sigma = torch.ones(pred.shape[0], pred.shape[1], device=pred.device)
                return pred, sigma

        return PointDeformNetV1()

    # V2: multi-scale encoder + per-vertex uncertainty head
    class PointDeformNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            # Three-level hierarchical encoder; each level's max-pool contributes to latent
            self.enc1 = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.enc3 = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
            # Aggregate multi-scale global features → single latent vector
            self.agg = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim + latent_dim, latent_dim), nn.ReLU()
            )
            # Shared decoder trunk (latent + template vertex position)
            self.dec_trunk = nn.Sequential(
                nn.Linear(latent_dim + 3, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.head_disp = nn.Linear(hidden_dim, 3)        # displacement per vertex
            self.head_sigma = nn.Linear(hidden_dim, 1)       # log-std per vertex

        def _encode(self, points):
            f1 = self.enc1(points)                            # (B, N, hidden)
            f2 = self.enc2(f1)                               # (B, N, hidden)
            f3 = self.enc3(f2)                               # (B, N, latent)
            g = torch.cat([f1.max(1).values,
                           f2.max(1).values,
                           f3.max(1).values], dim=-1)        # (B, hidden+hidden+latent)
            return self.agg(g)                               # (B, latent_dim)

        def forward(self, points, template_vertices):
            latent = self._encode(points)                    # (B, latent_dim)
            tpl_exp = template_vertices[None].expand(points.shape[0], -1, -1)
            lat_exp = latent[:, None, :].expand(-1, template_vertices.shape[0], -1)
            trunk = self.dec_trunk(torch.cat([tpl_exp, lat_exp], dim=-1))
            return tpl_exp + self.head_disp(trunk)

        def predict_with_uncertainty(self, points, template_vertices):
            latent = self._encode(points)
            tpl_exp = template_vertices[None].expand(points.shape[0], -1, -1)
            lat_exp = latent[:, None, :].expand(-1, template_vertices.shape[0], -1)
            trunk = self.dec_trunk(torch.cat([tpl_exp, lat_exp], dim=-1))
            pred = tpl_exp + self.head_disp(trunk)
            log_sigma = self.head_sigma(trunk).squeeze(-1)   # (B, V)
            sigma = torch.exp(log_sigma).clamp(0.01, 10.0)
            return pred, sigma

    if version == 2:
        return PointDeformNetV2()

    # V3: GNN-DeformNet — two-stream PointNet + GCN over template topology
    if template_faces is None:
        raise ValueError("template_faces required for model_version=3")
    adj_norm_cpu = build_gcn_adj(template_faces, vertex_count)

    class GCNLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, act: bool = True):
            super().__init__()
            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.use_act = act

        def forward(self, x: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
            out = torch.sparse.mm(adj, self.W(x))
            return F.relu(out) if self.use_act else out

    class PointDeformNetV3(nn.Module):
        def __init__(self):
            super().__init__()
            # Stream A: PointNet multi-scale encoder (same as V2)
            self.enc1 = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.enc3 = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU())
            self.agg = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim + latent_dim, latent_dim), nn.ReLU()
            )
            # Stream B: GCN over template mesh topology
            self.gcn1 = GCNLayer(3, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
            self.gcn3 = GCNLayer(hidden_dim, latent_dim, act=False)
            # Merge trunk + heads
            self.dec_trunk = nn.Sequential(
                nn.Linear(latent_dim + latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.head_disp = nn.Linear(hidden_dim, 3)
            self.head_sigma = nn.Linear(hidden_dim, 1)
            # Register normalised adjacency as a non-parameter buffer (moves with .to(device))
            self.register_buffer("adj_norm", adj_norm_cpu)

        def _encode_source(self, points: "torch.Tensor") -> "torch.Tensor":
            f1 = self.enc1(points)
            f2 = self.enc2(f1)
            f3 = self.enc3(f2)
            g = torch.cat([f1.max(1).values, f2.max(1).values, f3.max(1).values], dim=-1)
            return self.agg(g)                                    # (B, latent_dim)

        def _encode_template(self, tpl: "torch.Tensor") -> "torch.Tensor":
            adj = self.adj_norm.to(tpl.device)
            h1 = self.gcn1(tpl, adj)
            h2 = self.gcn2(h1, adj)
            return self.gcn3(h2, adj)                             # (V, latent_dim)

        def _decode(self, z_src: "torch.Tensor", g_tpl: "torch.Tensor", tpl: "torch.Tensor"):
            V = tpl.shape[0]
            B = z_src.shape[0]
            z_exp = z_src[:, None, :].expand(B, V, -1)           # (B, V, latent_dim)
            g_exp = g_tpl[None, :, :].expand(B, V, -1)           # (B, V, latent_dim)
            feat = torch.cat([z_exp, g_exp], dim=-1)              # (B, V, 2*latent_dim)
            trunk = self.dec_trunk(feat)
            return trunk

        def forward(self, points: "torch.Tensor", template_vertices: "torch.Tensor") -> "torch.Tensor":
            z = self._encode_source(points)
            g = self._encode_template(template_vertices)
            tpl = template_vertices[None].expand(points.shape[0], -1, -1)
            trunk = self._decode(z, g, template_vertices)
            return tpl + self.head_disp(trunk)

        def predict_with_uncertainty(self, points: "torch.Tensor", template_vertices: "torch.Tensor"):
            z = self._encode_source(points)
            g = self._encode_template(template_vertices)
            tpl = template_vertices[None].expand(points.shape[0], -1, -1)
            trunk = self._decode(z, g, template_vertices)
            pred = tpl + self.head_disp(trunk)
            log_sigma = self.head_sigma(trunk).squeeze(-1)        # (B, V)
            sigma = torch.exp(log_sigma).clamp(0.01, 10.0)
            return pred, sigma

    return PointDeformNetV3()


def edge_index_from_faces(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces.astype(int):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return np.asarray(sorted(edges), dtype=np.int64)


def laplacian_loss(pred, edges):
    if edges.numel() == 0:
        return pred.new_tensor(0.0)
    return (pred[:, edges[:, 0]] - pred[:, edges[:, 1]]).pow(2).mean()


def chamfer_torch(pred, target):
    torch, _, _, _ = _torch()
    dist = torch.cdist(pred, target)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def train_deformnet(config: TrainConfig) -> dict[str, object]:
    torch, _, DataLoader, _ = _torch()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device(config.device)
    template = load_template(config.template)
    records = read_pairs_manifest(config.pairs)
    if config.max_pairs is not None:
        records = records[: config.max_pairs]
    if not records:
        raise ValueError("No training pairs found")

    dataset = DeformNetDataset(records, template.mesh.vertex_count)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = build_model(
        template.mesh.vertex_count, config.latent_dim, config.hidden_dim,
        version=getattr(config, "model_version", 2),
        template_faces=template.mesh.faces if getattr(config, "model_version", 2) == 3 else None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    template_vertices = torch.tensor(template.mesh.vertices.astype(np.float32), device=device)
    edges = torch.tensor(edge_index_from_faces(template.mesh.faces), dtype=torch.long, device=device)

    # Landmark indices for landmark-consistency loss.
    lm_indices = getattr(template, "landmark_indices", None)
    lm_tensor = torch.tensor(lm_indices.astype(np.int64), device=device) if lm_indices is not None and len(lm_indices) > 0 else None
    use_lm_loss = lm_tensor is not None and getattr(config, "landmark_weight", 0.0) > 0.0
    use_uncertainty_loss = hasattr(model, "predict_with_uncertainty") and getattr(config, "model_version", 2) in (2, 3)

    history: list[dict[str, float]] = []
    start = perf_counter()
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for points, target, _ in loader:
            points = points.to(device)
            target = target.to(device)

            if use_uncertainty_loss:
                pred, sigma = model.predict_with_uncertainty(points, template_vertices)
                # Heteroscedastic NLL loss: per-vertex uncertainty-weighted reconstruction
                sq_err = (pred - target).pow(2).sum(-1)                    # (B, V)
                nll = 0.5 * (sq_err / sigma.pow(2) + 2.0 * sigma.log()).mean()
                loss = nll
            else:
                pred = model(points, template_vertices)
                l1 = (pred - target).abs().mean()
                l2 = (pred - target).pow(2).mean()
                loss = config.l1_weight * l1 + config.l2_weight * l2

            smooth = laplacian_loss(pred - template_vertices[None, :, :], edges)
            loss = loss + (config.edge_weight + config.laplacian_weight) * smooth

            if config.chamfer_weight > 0:
                loss = loss + config.chamfer_weight * chamfer_torch(pred, target)

            if use_lm_loss and lm_tensor is not None:
                lm_loss = (pred[:, lm_tensor] - target[:, lm_tensor]).pow(2).mean()
                loss = loss + config.landmark_weight * lm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})

    checkpoint_path = out / "deformnet.pt"
    model_version = getattr(config, "model_version", 2)
    ckpt_data = {
        "model_state": model.state_dict(),
        "train_config": asdict(config),
        "template_vertices": template.mesh.vertices.astype(np.float32),
        "template_faces": template.mesh.faces.astype(np.int64),
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "model_version": model_version,
    }
    if model_version == 3:
        ckpt_data["edge_index"] = edge_index_from_faces(template.mesh.faces)
    torch.save(ckpt_data, checkpoint_path)
    sample_paths = _write_sample_prediction(model, records[0], template.mesh, template_vertices, device, out)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "pair_count": len(records),
        "epochs": config.epochs,
        "history": history,
        "runtime_sec": perf_counter() - start,
        "sample_outputs": sample_paths,
    }
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _write_sample_prediction(model, record: PairRecord, template_mesh: Mesh, template_vertices, device, out: Path) -> dict[str, str]:
    torch, _, _, _ = _torch()
    model.eval()
    points = torch.tensor(np.load(record.source_points)["points"][None].astype(np.float32), device=device)
    with torch.no_grad():
        pred = model(points, template_vertices)[0].detach().cpu().numpy()
    pred_mesh = template_mesh.copy(vertices=pred, name="deformnet_prediction")
    target_mesh = load_mesh(record.target_mesh).copy(name="target_flame")
    source_mesh = load_mesh(record.source_mesh).copy(name="source")
    pred_path = out / "sample_prediction.obj"
    target_path = out / "sample_target.obj"
    viewer_path = out / "sample_viewer.html"
    write_obj(pred_mesh, pred_path)
    write_obj(target_mesh, target_path)
    metrics = {
        "sample_pair": record.pair_id,
        "pred_to_target_chamfer": chamfer_distance(pred_mesh, target_mesh),
        "pred_to_target_normal_consistency": normal_consistency(target_mesh, pred_mesh),
    }
    write_viewer(viewer_path, "DeformNet Sample Prediction", [source_mesh, template_mesh, pred_mesh, target_mesh], metrics)
    return {"prediction": str(pred_path), "target": str(target_path), "viewer": str(viewer_path)}


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    torch, _, _, _ = _torch()
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    version = int(ckpt.get("model_version", 2))
    template_faces = np.asarray(ckpt["template_faces"], dtype=np.int64)
    model = build_model(
        int(ckpt["template_vertices"].shape[0]),
        int(ckpt["latent_dim"]),
        int(ckpt["hidden_dim"]),
        version=version,
        template_faces=template_faces if version == 3 else None,
    )
    model.load_state_dict(ckpt["model_state"])
    template_mesh = Mesh(np.asarray(ckpt["template_vertices"], dtype=np.float64), template_faces, name="checkpoint_template")
    return model, template_mesh, ckpt
