from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is in requirements.txt but optional for non-learning commands
    torch = None


def _square_faces() -> np.ndarray:
    """Two triangles sharing the 0-2 diagonal (4 vertices, 5 unique edges)."""
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _dense_reference_adj(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """D^{-1/2}(A+I)D^{-1/2} computed densely, as an independent check of build_gcn_adj."""
    a = np.eye(vertex_count, dtype=np.float64)
    for f in faces:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            a[u, v] = a[v, u] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(a.sum(axis=1))
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


@unittest.skipIf(torch is None, "PyTorch is required for GNN DeformNet (v3) tests")
class GCNAdjacencyTests(unittest.TestCase):
    def test_matches_dense_symmetric_normalisation(self) -> None:
        from face_semantic_icp.train import build_gcn_adj

        faces = _square_faces()
        adj = build_gcn_adj(faces, 4).to_dense().numpy()
        np.testing.assert_allclose(adj, _dense_reference_adj(faces, 4), atol=1e-6)
        np.testing.assert_allclose(adj, adj.T, atol=1e-7)

    def test_shared_edges_are_not_double_counted(self) -> None:
        from face_semantic_icp.train import build_gcn_adj

        # Edge 0-2 appears in both triangles; it must still contribute weight 1, not 2.
        adj = build_gcn_adj(_square_faces(), 4).to_dense().numpy()
        # deg(0) = deg(2) = 4 (self + 3 neighbours) -> entry = 1/sqrt(4*4)
        self.assertAlmostEqual(float(adj[0, 2]), 0.25, places=6)

    def test_isolated_vertex_keeps_self_loop_without_warnings(self) -> None:
        from face_semantic_icp.train import build_gcn_adj

        # Vertex 4 is not referenced by any face; the self-loop keeps it well defined.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message="Sparse invariant checks")
            adj = build_gcn_adj(_square_faces(), 5).to_dense().numpy()
        self.assertTrue(np.isfinite(adj).all())
        self.assertAlmostEqual(float(adj[4, 4]), 1.0, places=6)
        self.assertEqual(float(np.abs(adj[4, :4]).sum()), 0.0)


@unittest.skipIf(torch is None, "PyTorch is required for GNN DeformNet (v3) tests")
class GNNDeformNetV3Tests(unittest.TestCase):
    def _model(self):
        from face_semantic_icp.train import build_model

        torch.manual_seed(0)
        return build_model(4, latent_dim=8, hidden_dim=16, version=3, template_faces=_square_faces()).eval()

    def test_requires_template_faces(self) -> None:
        from face_semantic_icp.train import build_model

        with self.assertRaises(ValueError):
            build_model(4, latent_dim=8, hidden_dim=16, version=3, template_faces=None)

    def test_forward_matches_uncertainty_prediction_and_sigma_is_clamped(self) -> None:
        model = self._model()
        gen = torch.Generator().manual_seed(1)
        points = torch.randn((2, 10, 3), generator=gen)
        template_vertices = torch.randn((4, 3), generator=gen)
        with torch.no_grad():
            pred = model(points, template_vertices)
            pred_u, sigma = model.predict_with_uncertainty(points, template_vertices)
        torch.testing.assert_close(pred, pred_u)
        self.assertTrue(bool((sigma >= 0.01).all()) and bool((sigma <= 10.0).all()))

    def test_prediction_is_invariant_to_source_point_order(self) -> None:
        # The PointNet stream max-pools over points, so shuffling the source must not change the output.
        model = self._model()
        gen = torch.Generator().manual_seed(2)
        points = torch.randn((1, 12, 3), generator=gen)
        template_vertices = torch.randn((4, 3), generator=gen)
        perm = torch.randperm(12, generator=gen)
        with torch.no_grad():
            a = model(points, template_vertices)
            b = model(points[:, perm], template_vertices)
        torch.testing.assert_close(a, b)

    def test_train_checkpoint_roundtrip_restores_v3(self) -> None:
        from face_semantic_icp.datasets import scan_dataset
        from face_semantic_icp.demo_data import make_random_topology_face
        from face_semantic_icp.geometry import write_obj
        from face_semantic_icp.pairs import generate_pairs
        from face_semantic_icp.train import TrainConfig, load_checkpoint, train_deformnet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generic_dataset" / "subject_01"
            root.mkdir(parents=True)
            write_obj(make_random_topology_face(seed=7, point_count=240), root / "scan.obj")
            scan_dataset("generic", Path(tmp) / "generic_dataset", Path(tmp) / "dataset_out")
            pairs_out = Path(tmp) / "pairs"
            generate_pairs(Path(tmp) / "dataset_out" / "dataset_manifest.jsonl", pairs_out, point_count=64, limit=1)

            torch.manual_seed(0)
            metrics = train_deformnet(
                TrainConfig(
                    pairs=str(pairs_out / "pairs_manifest.jsonl"),
                    out=str(Path(tmp) / "train_v3"),
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    latent_dim=8,
                    hidden_dim=16,
                    max_pairs=1,
                    model_version=3,
                )
            )
            model, template_mesh, ckpt = load_checkpoint(metrics["checkpoint"])

            self.assertEqual(ckpt["model_version"], 3)
            self.assertIn("edge_index", ckpt)
            self.assertIn("adj_norm", ckpt["model_state"])
            self.assertGreater(metrics["num_parameters"], 0)

            # The reloaded model must reproduce the sample prediction written at the end of training.
            from face_semantic_icp.geometry import load_mesh

            record_points = np.load(pairs_out / "pair_000000" / "source_points.npz")["points"].astype(np.float32)
            template_vertices = torch.tensor(template_mesh.vertices.astype(np.float32))
            with torch.no_grad():
                pred = model.eval()(torch.tensor(record_points[None]), template_vertices)[0].numpy()
            saved = load_mesh(metrics["sample_outputs"]["prediction"]).vertices
            np.testing.assert_allclose(pred, saved, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
