from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_semantic_icp.geometry import Mesh, load_mesh, vertex_normals, write_obj, write_ply
from face_semantic_icp.metrics import chamfer_distance, normal_consistency


def _read_pairs(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_hard_pair(records: list[dict[str, object]], mean_vertices: np.ndarray, max_scan: int) -> dict[str, object]:
    best_record = records[0]
    best_score = -1.0
    for record in records[:max_scan]:
        target = np.load(record["target_vertices"]).astype(np.float64)
        score = float(np.linalg.norm(target - mean_vertices, axis=1).mean())
        if score > best_score:
            best_score = score
            best_record = record
    return best_record


def _make_partial_observation(target: np.ndarray, keep_ratio: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = target[:, 0]
    y = target[:, 1]
    z = target[:, 2]

    # Front/visible face band + one side occlusion, roughly like an incomplete scan.
    front = z >= np.quantile(z, 0.28)
    central = np.abs(x) <= np.quantile(np.abs(x), 0.88)
    no_lower_back = y >= np.quantile(y, 0.10)
    side_dropout = ~((x > np.quantile(x, 0.55)) & (y < np.quantile(y, 0.60)))
    mask = front & central & no_lower_back & side_dropout
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        raise ValueError("Partial-observation mask selected no vertices; the target mesh may be degenerate or mis-oriented")
    keep = max(128, int(len(target) * keep_ratio))
    keep = min(keep, len(candidates))
    observed_idx = np.sort(rng.choice(candidates, size=keep, replace=False))
    points = target[observed_idx].copy()
    points += rng.normal(scale=noise, size=points.shape)
    return points.astype(np.float64), observed_idx.astype(np.int64)


def _fit_prior_to_points(
    points: np.ndarray,
    mean_vertices: np.ndarray,
    modes: np.ndarray,
    variance_ratio: np.ndarray,
    iterations: int,
    ridge: float,
    distance_gate: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    coeff = np.zeros(modes.shape[0], dtype=np.float64)
    mode_matrix = modes.reshape(modes.shape[0], -1)
    history: list[dict[str, float]] = []
    correspondences = np.zeros(len(points), dtype=np.int64)

    for step in range(iterations):
        current = mean_vertices + np.tensordot(coeff, modes, axes=(0, 0))
        tree = cKDTree(current)
        dist, idx = tree.query(points, k=1)
        gate = max(float(np.quantile(dist, 0.92)), distance_gate)
        accepted = dist <= gate
        if accepted.sum() < max(64, modes.shape[0] * 4):
            accepted = np.ones_like(accepted, dtype=bool)

        idx_acc = idx[accepted]
        points_acc = points[accepted]
        mean_acc = mean_vertices[idx_acc]
        a = modes[:, idx_acc, :].transpose(1, 2, 0).reshape(-1, modes.shape[0])
        b = (points_acc - mean_acc).reshape(-1)

        # Prior-weighted ridge: high-variance modes are allowed to move more.
        prior_diag = ridge / np.maximum(variance_ratio[: modes.shape[0]], 1e-6)
        lhs = a.T @ a + np.diag(prior_diag)
        rhs = a.T @ b
        coeff = np.linalg.solve(lhs, rhs)
        correspondences = idx
        history.append(
            {
                "iteration": float(step + 1),
                "accepted_ratio": float(accepted.mean()),
                "mean_nn_distance": float(dist.mean()),
                "median_nn_distance": float(np.median(dist)),
                "coefficient_norm": float(np.linalg.norm(coeff)),
            }
        )

    completed = mean_vertices + np.tensordot(coeff, modes, axes=(0, 0))
    return completed, correspondences, history


def _vertex_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _point_to_mesh_distance(points: np.ndarray, vertices: np.ndarray) -> float:
    dist, _ = cKDTree(vertices).query(points, k=1)
    return float(dist.mean())


def _render_report(out: Path, points: np.ndarray, mean_mesh: Mesh, completed_mesh: Mesh, target_mesh: Mesh, metrics: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import FancyBboxPatch

    bg = "#f8fafc"
    blue = "#0d47a1"
    text = "#1f2937"
    muted = "#64748b"
    border = "#cbd5e1"
    point_color = "#f59e0b"
    colors = [np.array([0.58, 0.66, 0.76]), np.array([0.36, 0.74, 0.56]), np.array([0.50, 0.68, 0.86])]
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "font.family": "DejaVu Sans",
        "text.color": text,
    })

    def centered(v: np.ndarray) -> np.ndarray:
        mn, mx = v.min(axis=0), v.max(axis=0)
        return (v - (mn + mx) * 0.5) / max(float((mx - mn).max()), 1e-9)

    def render_points(ax, pts: np.ndarray, title: str, subtitle: str) -> None:
        v = centered(pts)
        order = np.argsort(v[:, 2])
        ax.scatter(v[order, 0] * 0.86 + 0.5, v[order, 1] * 0.86 + 0.5, s=8, c=point_color, alpha=0.78, edgecolors="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=blue, fontsize=13, fontweight="bold")
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center", va="top", color=muted, fontsize=9)

    def render_mesh(ax, mesh: Mesh, color: np.ndarray, title: str, subtitle: str, heat: np.ndarray | None = None) -> None:
        verts = centered(mesh.vertices)
        faces = mesh.faces
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        depth = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
        order = np.argsort(depth)
        fs = faces[order]
        pts = np.stack([np.stack([verts[fs[:, k], 0], verts[fs[:, k], 1]], axis=-1) for k in range(3)], axis=1) * 0.86 + 0.5
        light = np.array([0.25, 0.55, 0.80])
        light /= np.linalg.norm(light)
        shade = np.clip(normals @ light, 0.35, 1.0)[order]
        if heat is None:
            fc = np.clip(color[None, :] * (0.70 + 0.35 * shade[:, None]), 0, 1)
        else:
            h = np.clip(heat[fs].mean(axis=1), 0, 1)
            c0, c1, c2 = np.array([0.12, 0.35, 0.85]), np.array([0.94, 0.96, 0.98]), np.array([0.90, 0.18, 0.16])
            fc = np.where(h[:, None] < 0.5, c0 + (c1 - c0) * (h[:, None] * 2.0), c1 + (c2 - c1) * ((h[:, None] - 0.5) * 2.0))
        ax.add_collection(PolyCollection(pts, facecolors=np.c_[fc, np.ones(len(fc))], edgecolors=(1, 1, 1, 0.04), linewidths=0.02))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=blue, fontsize=13, fontweight="bold")
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center", va="top", color=muted, fontsize=9)

    fig = plt.figure(figsize=(18, 10))
    fig.text(0.035, 0.96, "New Real Method: Prior-Guided FLAME Completion", color=blue, fontsize=27, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.91, "A partial/noisy real FaMoS observation is completed into a full ordered FLAME mesh using the learned deformation prior.", color=text, fontsize=13, ha="left", va="top")

    axes = [
        fig.add_axes([0.04, 0.46, 0.20, 0.36]),
        fig.add_axes([0.28, 0.46, 0.20, 0.36]),
        fig.add_axes([0.52, 0.46, 0.20, 0.36]),
        fig.add_axes([0.76, 0.46, 0.20, 0.36]),
    ]
    err = np.linalg.norm(completed_mesh.vertices - target_mesh.vertices, axis=1)
    heat = err / max(float(np.percentile(err, 95)), 1e-9)
    render_points(axes[0], points, "Input", f"{len(points)} noisy partial points")
    render_mesh(axes[1], mean_mesh, colors[0], "Baseline", f"mean prior Chamfer {metrics['baseline_chamfer']:.5f}")
    render_mesh(axes[2], completed_mesh, colors[1], "Prior-Guided Output", f"completed Chamfer {metrics['completed_chamfer']:.5f}")
    render_mesh(axes[3], completed_mesh, colors[2], "Residual Heatmap", f"vertex RMSE {metrics['completed_vertex_rmse']:.5f}", heat=heat)

    ax = fig.add_axes([0.05, 0.09, 0.90, 0.27])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#ffffff", edgecolor=border, linewidth=1.5))
    ax.text(0.025, 0.82, "What is novel here?", color=blue, fontsize=16, fontweight="bold", va="top")
    ax.text(0.025, 0.62, "This is no longer only a visualization. A real FaMoS-learned prior is actively used to complete missing/noisy facial geometry into a full fixed-topology FLAME mesh.", color=text, fontsize=12.2, va="top", wrap=True)
    ax.text(0.025, 0.34, f"Measured gain: Chamfer {metrics['baseline_chamfer']:.5f} -> {metrics['completed_chamfer']:.5f}; vertex RMSE {metrics['baseline_vertex_rmse']:.5f} -> {metrics['completed_vertex_rmse']:.5f}; runtime {metrics['runtime_sec']:.3f}s.", color="#059669", fontsize=12.2, fontweight="bold", va="top")
    ax.text(0.025, 0.16, "Meeting sentence: We added prior-guided completion, where the learned FaMoS deformation space is used as an active shape prior for fixed-topology reconstruction from incomplete observations.", color=blue, fontsize=12.2, fontweight="bold", va="top", wrap=True)
    fig.savefig(out / "01_prior_guided_completion_showcase.png", dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run prior-guided FLAME completion on a real FaMoS registration stress case.")
    parser.add_argument("--prior", default="outputs/famos_deformation_prior/famos_deformation_prior.npz")
    parser.add_argument("--pairs", default="outputs/famos_flame_subset_pairs/pairs_manifest.jsonl")
    parser.add_argument("--out", default="outputs/famos_prior_guided_completion")
    parser.add_argument("--pair-id", default="hardest", help="Pair id such as pair_000012, or 'hardest'.")
    parser.add_argument("--scan-limit", type=int, default=1000)
    parser.add_argument("--keep-ratio", type=float, default=0.34)
    parser.add_argument("--noise", type=float, default=0.012)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=0.020)
    parser.add_argument("--distance-gate", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    if not 0.0 < args.keep_ratio <= 1.0:
        parser.error(f"--keep-ratio must be in (0, 1], got {args.keep_ratio}")
    if args.noise < 0.0:
        parser.error(f"--noise must be non-negative, got {args.noise}")
    if args.iterations < 1:
        parser.error(f"--iterations must be at least 1, got {args.iterations}")
    if args.ridge < 0.0:
        parser.error(f"--ridge must be non-negative, got {args.ridge}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prior = np.load(args.prior)
    mean_vertices = prior["mean_vertices"].astype(np.float64)
    modes = prior["modes"].astype(np.float64)
    variance_ratio = prior["explained_variance_ratio"].astype(np.float64)
    faces = prior["faces"].astype(np.int64)
    records = _read_pairs(Path(args.pairs))
    if args.pair_id == "hardest":
        record = _select_hard_pair(records, mean_vertices, args.scan_limit)
    else:
        matches = [r for r in records if r["pair_id"] == args.pair_id]
        if not matches:
            raise SystemExit(f"Pair id not found: {args.pair_id}")
        record = matches[0]

    target_vertices = np.load(record["target_vertices"]).astype(np.float64)
    target_mesh = Mesh(target_vertices, faces, name="real_famos_target")
    points, observed_idx = _make_partial_observation(target_vertices, args.keep_ratio, args.noise, args.seed)

    start = perf_counter()
    completed_vertices, correspondences, history = _fit_prior_to_points(
        points,
        mean_vertices,
        modes,
        variance_ratio,
        iterations=args.iterations,
        ridge=args.ridge,
        distance_gate=args.distance_gate,
    )
    runtime = perf_counter() - start

    mean_mesh = Mesh(mean_vertices, faces, name="prior_mean_baseline")
    completed_mesh = Mesh(completed_vertices, faces, name="prior_guided_completion")

    metrics = {
        "method": "Prior-Guided FLAME Completion",
        "pair_id": record["pair_id"],
        "source_item_id": record.get("source_item_id"),
        "observed_points": int(len(points)),
        "total_vertices": int(target_vertices.shape[0]),
        "keep_ratio": float(len(points) / target_vertices.shape[0]),
        "noise": float(args.noise),
        "baseline_chamfer": float(chamfer_distance(mean_mesh, target_mesh)),
        "completed_chamfer": float(chamfer_distance(completed_mesh, target_mesh)),
        "baseline_vertex_rmse": _vertex_rmse(mean_vertices, target_vertices),
        "completed_vertex_rmse": _vertex_rmse(completed_vertices, target_vertices),
        "observed_to_baseline_distance": _point_to_mesh_distance(points, mean_vertices),
        "observed_to_completed_distance": _point_to_mesh_distance(points, completed_vertices),
        "normal_consistency_completed": float(normal_consistency(completed_mesh, target_mesh)),
        "runtime_sec": float(runtime),
        "iterations": int(args.iterations),
        "ridge": float(args.ridge),
        "history": history,
    }

    write_obj(Mesh(points, np.zeros((0, 3), dtype=np.int64), name="partial_observation_points"), out / "partial_observation_points.obj")
    write_obj(mean_mesh, out / "baseline_prior_mean.obj")
    write_obj(completed_mesh, out / "prior_guided_completion.obj")
    write_ply(completed_mesh, out / "prior_guided_completion.ply")
    write_obj(target_mesh, out / "real_famos_target.obj")
    np.save(out / "observed_indices.npy", observed_idx)
    np.save(out / "correspondences.npy", correspondences)
    (out / "completion_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _render_report(out, points, mean_mesh, completed_mesh, target_mesh, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
