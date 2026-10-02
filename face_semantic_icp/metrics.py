from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Mesh, topology_signature, vertex_normals


def chamfer_distance(a: Mesh, b: Mesh) -> float:
    ta = cKDTree(a.vertices)
    tb = cKDTree(b.vertices)
    da, _ = tb.query(a.vertices, k=1)
    db, _ = ta.query(b.vertices, k=1)
    return float(0.5 * (da.mean() + db.mean()))


def hausdorff_distance(a: Mesh, b: Mesh) -> float:
    """One-sided and symmetric Hausdorff distance between two meshes.

    Unlike Chamfer distance (which averages all nearest-neighbour distances),
    Hausdorff captures the *worst-case* deviation — useful for spotting local
    surface errors that the mean would otherwise hide.

    Returns the symmetric Hausdorff distance: max(directed_a→b, directed_b→a).
    """
    ta = cKDTree(a.vertices)
    tb = cKDTree(b.vertices)
    da, _ = tb.query(a.vertices, k=1)
    db, _ = ta.query(b.vertices, k=1)
    return float(max(da.max(), db.max()))


def mean_edge_length(mesh: Mesh) -> float:
    """Average edge length across all triangle edges in the mesh.

    Useful as a normalisation factor when comparing Chamfer / Hausdorff
    distances across meshes of different scales.
    """
    v = mesh.vertices
    f = mesh.faces
    e0 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    e1 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
    e2 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
    return float(np.concatenate([e0, e1, e2]).mean())


def normal_consistency(a: Mesh, b: Mesh) -> float:
    ta = cKDTree(a.vertices)
    _, idx = ta.query(b.vertices, k=1)
    na = vertex_normals(a)
    nb = vertex_normals(b)
    dots = np.abs(np.sum(na[idx] * nb, axis=1))
    return float(np.clip(dots.mean(), 0.0, 1.0))


def landmark_rmse(mesh: Mesh, landmark_indices: np.ndarray, target_landmarks: np.ndarray) -> float:
    if target_landmarks is None or len(target_landmarks) == 0:
        return float("nan")
    pred = mesh.vertices[landmark_indices]
    n = min(len(pred), len(target_landmarks))
    return float(np.sqrt(np.mean(np.sum((pred[:n] - target_landmarks[:n]) ** 2, axis=1))))


def semantic_chamfer(source: Mesh, result: Mesh, regions: list[str]) -> dict[str, float]:
    if source.semantics is None or result.semantics is None:
        return {}
    values: dict[str, float] = {}
    for region in regions:
        sm = source.copy(vertices=source.vertices[source.semantics == region], faces=np.zeros((0, 3), dtype=np.int64))
        rm = result.copy(vertices=result.vertices[result.semantics == region], faces=np.zeros((0, 3), dtype=np.int64))
        if sm.vertex_count == 0 or rm.vertex_count == 0:
            continue
        values[region] = chamfer_distance(sm, rm)
    return values


def build_metrics(source: Mesh, initial: Mesh, wrapped: Mesh, detailed: Mesh, landmark_indices: np.ndarray, target_landmarks: np.ndarray | None, timings: dict[str, float]) -> dict[str, object]:
    return {
        "method": "Face Semantic ICP Wrap",
        "chamfer_initial_to_source": chamfer_distance(initial, source),
        "chamfer_wrapped_to_source": chamfer_distance(wrapped, source),
        "chamfer_detailed_to_source": chamfer_distance(detailed, source),
        "normal_consistency_wrapped": normal_consistency(source, wrapped),
        "normal_consistency_detailed": normal_consistency(source, detailed),
        "landmark_rmse": landmark_rmse(wrapped, landmark_indices, target_landmarks) if target_landmarks is not None else None,
        "semantic_region_chamfer": semantic_chamfer(source, detailed, ["nose", "eyes", "lips", "jaw"]),
        "topology": topology_signature(detailed),
        "timings_sec": timings,
    }

