from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_semantic_icp.geometry import Mesh, load_mesh, write_obj


def _load_records(path: Path, limit: int | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
        if limit is not None and len(records) >= limit:
            break
    return records


def _render_prior_report(out: Path, mean: np.ndarray, modes: np.ndarray, variance: np.ndarray, faces: np.ndarray, count: int) -> None:
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
    colors = [
        np.array([0.50, 0.68, 0.86]),
        np.array([0.40, 0.74, 0.59]),
        np.array([0.85, 0.58, 0.38]),
        np.array([0.72, 0.58, 0.82]),
    ]
    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "font.family": "DejaVu Sans",
        "text.color": text,
    })

    def centered(vertices: np.ndarray) -> np.ndarray:
        mn = vertices.min(axis=0)
        mx = vertices.max(axis=0)
        return (vertices - (mn + mx) * 0.5) / max(float((mx - mn).max()), 1e-9)

    def render_mesh(ax, vertices: np.ndarray, color: np.ndarray, title: str, subtitle: str = "") -> None:
        verts = centered(vertices)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        depth = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
        order = np.argsort(depth)
        fs = faces[order]
        pts = np.stack(
            [np.stack([verts[fs[:, k], 0], verts[fs[:, k], 1]], axis=-1) for k in range(3)],
            axis=1,
        ) * 0.86 + 0.5
        light = np.array([0.25, 0.55, 0.80])
        light /= np.linalg.norm(light)
        shade = np.clip(normals @ light, 0.34, 1.0)[order]
        fc = np.clip(color[None, :] * (0.68 + 0.38 * shade[:, None]), 0, 1)
        ax.add_collection(PolyCollection(pts, facecolors=np.c_[fc, np.ones(len(fc))], edgecolors=(1, 1, 1, 0.04), linewidths=0.025))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=blue, fontsize=13, fontweight="bold")
        if subtitle:
            ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center", va="top", color=muted, fontsize=9)

    fig = plt.figure(figsize=(18, 10))
    fig.text(0.035, 0.955, "More Novel Piece: FaMoS-Learned Deformation Prior", color=blue, fontsize=27, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.905, "A real dataset prior learns common face-shape/expression deformation modes from FLAME registrations.", color=text, fontsize=13, ha="left", va="top")

    axes = [
        fig.add_axes([0.05, 0.49, 0.20, 0.34]),
        fig.add_axes([0.29, 0.49, 0.20, 0.34]),
        fig.add_axes([0.53, 0.49, 0.20, 0.34]),
        fig.add_axes([0.77, 0.49, 0.18, 0.34]),
    ]
    render_mesh(axes[0], mean, colors[0], "Mean FaMoS Face", f"{count} real registrations")
    for i in range(3):
        amp = 3.0 * float(np.sqrt(max(variance[i], 1e-12)))
        shape = mean + modes[i] * amp
        render_mesh(axes[i + 1], shape, colors[i + 1], f"Prior Mode {i + 1}", f"explained {variance[i] * 100:.2f}%")

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.28])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#ffffff", edgecolor=border, linewidth=1.5))
    ax.text(0.025, 0.82, "Why this is stronger novelty", color=blue, fontsize=16, fontweight="bold", va="top")
    ax.text(0.025, 0.62, "Instead of only fitting a template frame-by-frame, we learn a compact deformation space from real FaMoS registrations. This prior can regularize future raw-scan wrapping and gives DeformNet a dataset-backed shape/expression manifold.", color=text, fontsize=12.2, va="top", wrap=True)
    ax.text(0.025, 0.32, "Meeting sentence:", color=blue, fontsize=14, fontweight="bold", va="top")
    ax.text(0.025, 0.17, "The extra novelty is a real FaMoS-learned deformation prior: Wrap++ gives the teacher mechanism, FaMoS gives the supervised deformation space, and DeformNet learns a fast fixed-topology predictor.", color="#059669", fontsize=12.2, fontweight="bold", va="top", wrap=True)

    fig.savefig(out / "01_famos_deformation_prior_showcase.png", dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Learn a low-dimensional FLAME deformation prior from FaMoS training pairs.")
    parser.add_argument("--pairs", default="outputs/famos_flame_subset_pairs/pairs_manifest.jsonl")
    parser.add_argument("--out", default="outputs/famos_deformation_prior")
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--components", type=int, default=12)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.pairs), args.max_pairs)
    if not records:
        raise SystemExit("No pair records found")

    targets: list[np.ndarray] = []
    first_mesh = load_mesh(records[0]["target_mesh"])
    faces = first_mesh.faces.astype(np.int64)
    for record in records:
        targets.append(np.load(record["target_vertices"]).astype(np.float32))
    data = np.stack([v.reshape(-1) for v in targets], axis=0)
    mean_flat = data.mean(axis=0)
    centered = data - mean_flat[None, :]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(args.components, len(records)), svd_solver="randomized", random_state=7)
    pca.fit(centered)

    mean_vertices = mean_flat.reshape(targets[0].shape)
    modes = pca.components_.reshape(pca.n_components_, targets[0].shape[0], 3)
    variance = pca.explained_variance_ratio_.astype(np.float64)

    np.savez_compressed(
        out / "famos_deformation_prior.npz",
        mean_vertices=mean_vertices.astype(np.float32),
        modes=modes.astype(np.float32),
        explained_variance_ratio=variance.astype(np.float32),
        faces=faces.astype(np.int64),
        pair_count=np.asarray([len(records)], dtype=np.int64),
    )

    write_obj(Mesh(mean_vertices, faces, name="famos_prior_mean"), out / "famos_prior_mean.obj")
    for i in range(min(3, len(modes))):
        amp = 3.0 * float(np.sqrt(max(variance[i], 1e-12)))
        write_obj(Mesh(mean_vertices + modes[i] * amp, faces, name=f"famos_prior_mode_{i + 1}"), out / f"famos_prior_mode_{i + 1}.obj")

    summary = {
        "method": "FaMoS-learned FLAME deformation prior",
        "pair_count": len(records),
        "components": int(pca.n_components_),
        "topology": {"vertex_count": int(mean_vertices.shape[0]), "face_count": int(faces.shape[0])},
        "explained_variance_ratio": variance.tolist(),
        "cumulative_explained_variance": np.cumsum(variance).tolist(),
        "artifacts": {
            "prior": str(out / "famos_deformation_prior.npz"),
            "mean_obj": str(out / "famos_prior_mean.obj"),
            "showcase": str(out / "01_famos_deformation_prior_showcase.png"),
        },
    }
    (out / "prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _render_prior_report(out, mean_vertices, modes, variance, faces, len(records))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
