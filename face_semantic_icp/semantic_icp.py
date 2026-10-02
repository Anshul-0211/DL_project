from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .config import SemanticICPConfig, quality_schedule
from .deformation_prior import DeformationPrior
from .geometry import Mesh, build_cotangent_laplacian, build_vertex_adjacency, normalize_mesh, vertex_normals
from .template import TemplateSpec, classify_vertices, face_surface


REGION_MOBILITY = {
    "forehead": 0.70,
    "brows": 0.82,
    "eyes": 0.62,
    "nose": 0.55,
    "lips": 0.90,
    "cheeks": 1.00,
    "jaw": 0.85,
    "ears_neck": 0.45,
}

REGION_IMPORTANCE = {
    "forehead": 0.75,
    "brows": 1.35,
    "eyes": 1.75,
    "nose": 1.90,
    "lips": 1.85,
    "cheeks": 1.00,
    "jaw": 1.20,
    "ears_neck": 0.50,
}


@dataclass
class WrapResult:
    initial: Mesh
    wrapped: Mesh
    stage_logs: list[dict[str, float]]
    timings: dict[str, float]
    convergence_history: list[dict[str, float]] | None = None
    correspondence_stats: dict[str, float] | None = None


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / max(len(src), 1)
    u, s, vh = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vh
    var = np.mean(np.sum(src_c**2, axis=1))
    scale = float(np.sum(s * d) / max(var, 1e-12))
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def apply_similarity(vertices: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (vertices @ rotation.T) + translation


def estimate_source_landmarks(source: Mesh, landmark_uv: np.ndarray) -> np.ndarray:
    target = face_surface(landmark_uv[:, 0], landmark_uv[:, 1])
    target = normalize_points(target)
    source_vertices = normalize_points(source.vertices)
    tree = cKDTree(source_vertices)
    _, idx = tree.query(target, k=1)
    return source.vertices[idx]


def normalize_points(points: np.ndarray) -> np.ndarray:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) * 0.5
    extent = max(float(np.max(mx - mn)), 1e-12)
    return (points - center) / extent * 2.0


def initial_alignment(template: TemplateSpec, source: Mesh, source_landmarks: np.ndarray | None = None) -> Mesh:
    mesh = template.mesh
    if source_landmarks is not None and len(source_landmarks) >= len(template.landmark_indices):
        src = mesh.vertices[template.landmark_indices]
        dst = source_landmarks[: len(template.landmark_indices)]
        scale, rot, trans = umeyama_align(src, dst)
        vertices = apply_similarity(mesh.vertices, scale, rot, trans)
        return mesh.copy(vertices=vertices, name="initial_flame")

    tmn, tmx = mesh.bounds()
    smn, smx = source.bounds()
    tcenter = (tmn + tmx) * 0.5
    scenter = (smn + smx) * 0.5
    scale = float(np.max(smx - smn) / max(np.max(tmx - tmn), 1e-12))
    vertices = (mesh.vertices - tcenter) * scale + scenter
    return mesh.copy(vertices=vertices, name="initial_flame")


def build_laplacian(faces: np.ndarray, vertex_count: int) -> sparse.csr_matrix:
    neighbors = build_vertex_adjacency(faces, vertex_count)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, ns in enumerate(neighbors):
        rows.append(i)
        cols.append(i)
        data.append(1.0)
        if ns:
            w = -1.0 / len(ns)
            for j in ns:
                rows.append(i)
                cols.append(j)
                data.append(w)
    return sparse.csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))


def _trees_by_region(source: Mesh) -> dict[str, tuple[np.ndarray, cKDTree]]:
    labels = source.semantics if source.semantics is not None else classify_vertices(source)
    trees: dict[str, tuple[np.ndarray, cKDTree]] = {}
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) > 0:
            trees[str(label)] = (idx, cKDTree(source.vertices[idx]))
    return trees


def _visibility_scores(source: Mesh, source_idx: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones(len(source_idx), dtype=np.float64)
    normals = vertex_normals(source)                          # (V, 3) unit normals
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)   # canonical front-facing
    cos_vis = np.clip(normals[source_idx] @ view_dir, 0.0, 1.0)
    return 0.1 + 0.9 * cos_vis                               # range [0.1, 1.0]


def _arap_rotations(base_verts: np.ndarray, curr_verts: np.ndarray,
                    adjacency: list[list[int]]) -> np.ndarray:
    """Per-vertex best-fit rigid rotation mapping base neighborhood to current, via SVD."""
    R = np.empty((len(base_verts), 3, 3), dtype=np.float64)
    for i, nbrs in enumerate(adjacency):
        if not nbrs:
            R[i] = np.eye(3)
            continue
        nb = np.array(nbrs, dtype=np.int64)
        db = base_verts[nb] - base_verts[i]   # (k, 3) edge vectors in rest pose
        dc = curr_verts[nb] - curr_verts[i]   # (k, 3) edge vectors in current pose
        S = db.T @ dc                          # (3, 3) cross-covariance
        U, _, Vt = np.linalg.svd(S)
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:             # fix reflections
            Vt[-1] *= -1
            Ri = Vt.T @ U.T
        R[i] = Ri
    return R


def _arap_target_positions(base_verts: np.ndarray, curr_verts: np.ndarray,
                            adjacency: list[list[int]], R: np.ndarray) -> np.ndarray:
    """Rotation-consistent position estimates: each neighbor j votes for where i should be."""
    result = np.zeros_like(base_verts)
    counts = np.zeros(len(base_verts), dtype=np.float64)
    for i, nbrs in enumerate(adjacency):
        if not nbrs:
            result[i] = curr_verts[i]
            counts[i] = 1.0
            continue
        for j in nbrs:
            result[i] += curr_verts[j] + R[j] @ (base_verts[i] - base_verts[j])
            counts[i] += 1.0
    return result / np.maximum(counts[:, None], 1.0)


def semantic_correspondences(current: Mesh, source: Mesh, distance_gate: float, normal_gate_deg: float, config: SemanticICPConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    config = config or SemanticICPConfig()
    source_labels = source.semantics if source.semantics is not None else classify_vertices(source)
    current_labels = current.semantics if current.semantics is not None else classify_vertices(current)
    source_normals = vertex_normals(source)
    current_normals = vertex_normals(current)
    trees = _trees_by_region(source.copy(semantics=source_labels))
    all_tree = cKDTree(source.vertices)
    current_tree = cKDTree(current.vertices)
    cos_gate = float(np.cos(np.deg2rad(normal_gate_deg)))

    targets = current.vertices.copy()
    weights = np.zeros(current.vertex_count, dtype=np.float64)
    accepted = 0
    rejected_bidir = 0
    for region in np.unique(current_labels):
        ids = np.where(current_labels == region)[0]
        if len(ids) == 0:
            continue
        tree_info = trees.get(str(region)) if config.use_semantics else None
        if tree_info is None:
            dist, nn = all_tree.query(current.vertices[ids], k=1)
            src_idx = nn
        else:
            source_idx, tree = tree_info
            dist, local = tree.query(current.vertices[ids], k=1)
            src_idx = source_idx[local]
        normal_dot = np.sum(current_normals[ids] * source_normals[src_idx], axis=1)
        normal_agreement = np.abs(normal_dot)
        valid = (dist < distance_gate) & (normal_agreement > cos_gate)
        if config.bidirectional_check:
            reverse_dist, reverse_idx = current_tree.query(source.vertices[src_idx], k=1)
            bidir_valid = reverse_dist < max(distance_gate * 1.75, 1e-6)
            rejected_bidir += int(np.count_nonzero(valid & ~bidir_valid))
            valid = valid & bidir_valid
        confidence = np.exp(-((dist / max(distance_gate, 1e-9)) ** 2)) * np.clip((normal_agreement - cos_gate) / max(1.0 - cos_gate, 1e-9), 0.0, 1.0)
        confidence *= _visibility_scores(source, src_idx, config.visibility_weighting)
        importance = REGION_IMPORTANCE.get(str(region), 1.0)
        mobility = REGION_MOBILITY.get(str(region), 0.8)
        w = confidence * importance * mobility
        target_points = source.vertices[src_idx]
        plane_points = current.vertices[ids] + (np.sum((target_points - current.vertices[ids]) * source_normals[src_idx], axis=1))[:, None] * source_normals[src_idx]
        plane_ratio = float(np.clip(config.point_to_plane_ratio, 0.0, 1.0))
        mixed = (1.0 - plane_ratio) * target_points + plane_ratio * plane_points
        targets[ids[valid]] = mixed[valid]
        weights[ids[valid]] = w[valid]
        accepted += int(valid.sum())

    log = {
        "accepted_ratio": float(accepted / max(current.vertex_count, 1)),
        "mean_weight": float(weights[weights > 0].mean()) if np.any(weights > 0) else 0.0,
        "rejected_bidirectional": float(rejected_bidir),
        "semantic_enabled": float(bool(config.use_semantics)),
        "visibility_enabled": float(bool(config.visibility_weighting)),
    }
    return targets, weights, log


def solve_stage(base: Mesh, current: Mesh, source: Mesh, source_landmarks: np.ndarray | None, landmark_indices: np.ndarray, distance_gate: float, normal_gate_deg: float, smooth: float, corr_weight: float, landmark_weight: float, prior: float, ltl: sparse.csr_matrix, config: SemanticICPConfig | None = None, landmark_confidences: np.ndarray | None = None, adjacency: list[list[int]] | None = None, deformation_prior: "DeformationPrior | None" = None, uncertainty_sigma: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    config = config or SemanticICPConfig()
    targets, weights, log = semantic_correspondences(current, source, distance_gate, normal_gate_deg, config)
    n = current.vertex_count
    weights = weights * corr_weight

    # Uncertainty-guided weighting: scale correspondence weights by 1/sigma
    if uncertainty_sigma is not None and len(uncertainty_sigma) == n:
        inv_sigma = 1.0 / (uncertainty_sigma.astype(np.float64) + 1e-6)
        inv_sigma_mean = float(inv_sigma.mean())
        if inv_sigma_mean > 0:
            inv_sigma /= inv_sigma_mean
        weights = weights * inv_sigma

    # ARAP: compute rotation-consistent target positions and add as soft constraint
    aw = float(config.arap_weight)
    if aw > 0.0 and adjacency is not None:
        R = _arap_rotations(base.vertices, current.vertices, adjacency)
        arap_tgt = _arap_target_positions(base.vertices, current.vertices, adjacency, R)
    else:
        aw = 0.0
        arap_tgt = None

    # PCA prior: data-driven deformation subspace soft constraint
    pw = float(config.pca_weight) if deformation_prior is not None else 0.0
    pca_tgt: np.ndarray | None = None
    if pw > 0.0 and deformation_prior is not None:
        try:
            pca_tgt = deformation_prior.project(current.vertices)
        except Exception:
            pw = 0.0

    diag_vals = weights + prior + aw + pw
    a = sparse.diags(diag_vals, format="csr") + smooth * ltl
    rhs_base = (weights[:, None] * targets) + (prior * base.vertices) + smooth * (ltl @ base.vertices)
    if aw > 0.0 and arap_tgt is not None:
        rhs_base = rhs_base + aw * arap_tgt
    if pw > 0.0 and pca_tgt is not None:
        rhs_base = rhs_base + pw * pca_tgt

    if source_landmarks is not None and len(source_landmarks) > 0:
        lm_diag_values = np.zeros(n, dtype=np.float64)
        lm_targets = np.zeros((n, 3), dtype=np.float64)
        lm_ids = landmark_indices[: len(source_landmarks)]
        lm_weights = np.ones(len(lm_ids), dtype=np.float64)
        if landmark_confidences is not None and len(landmark_confidences) >= len(lm_ids):
            lm_weights = np.asarray(landmark_confidences[: len(lm_ids)], dtype=np.float64)
        lm_diag_values[lm_ids] = landmark_weight * lm_weights * config.feature_anchor_weight
        lm_targets[lm_ids] = source_landmarks[: len(lm_ids)]
        lm_diag = sparse.diags(lm_diag_values, format="csr")
        a = a + lm_diag
        rhs_base = rhs_base + lm_diag_values[:, None] * lm_targets

    vertices = np.zeros_like(current.vertices)
    for d in range(3):
        vertices[:, d] = spsolve(a, rhs_base[:, d])
    log.update({
        "distance_gate": distance_gate,
        "smooth": smooth,
        "corr_weight": corr_weight,
        "arap_weight": aw,
        "pca_weight": pw,
        "uncertainty_weighted": float(uncertainty_sigma is not None and len(uncertainty_sigma) == n),
    })
    return vertices, log


def run_semantic_icp(template: TemplateSpec, source: Mesh, source_landmarks: np.ndarray | None = None, config: SemanticICPConfig | None = None, landmark_confidences: np.ndarray | None = None, initial_vertices: np.ndarray | None = None, uncertainty_sigma: np.ndarray | None = None) -> WrapResult:
    config = config or SemanticICPConfig()
    start = perf_counter()
    source = normalize_mesh(source)
    if source.semantics is None:
        source = source.copy(semantics=classify_vertices(source))

    # If a warm-start is provided, use it; otherwise use Umeyama alignment
    if initial_vertices is not None and initial_vertices.shape == template.mesh.vertices.shape:
        initial = template.mesh.copy(vertices=initial_vertices, name="initial_flame")
    else:
        initial = initial_alignment(template, source, source_landmarks)
    align_time = perf_counter()
    current = initial.copy(name="semantic_icp_stage0")
    lap = build_cotangent_laplacian(initial)
    ltl = (lap.T @ lap).tocsr()

    # Precompute adjacency once for ARAP (reused across all stages)
    adjacency = build_vertex_adjacency(initial.faces, initial.vertex_count) if config.arap_weight > 0.0 else None

    # Load PCA prior once if configured
    deformation_prior: DeformationPrior | None = None
    if config.pca_weight > 0.0 and config.pca_prior_path:
        try:
            deformation_prior = DeformationPrior.load(config.pca_prior_path)
            if deformation_prior.vertex_count != initial.vertex_count:
                deformation_prior = None
        except Exception:
            deformation_prior = None

    schedule = config.schedule or quality_schedule(config.quality)
    logs: list[dict[str, float]] = []
    for stage_id, params in enumerate(schedule, start=1):
        vertices, log = solve_stage(
            initial,
            current,
            source,
            source_landmarks,
            template.landmark_indices,
            float(params["distance_gate"]),
            float(params["normal_gate_deg"]),
            float(params["smooth"]),
            float(params["corr_weight"]),
            float(params["landmark_weight"]),
            float(params["prior"]),
            ltl=ltl,
            config=config,
            landmark_confidences=landmark_confidences,
            adjacency=adjacency,
            deformation_prior=deformation_prior,
            uncertainty_sigma=uncertainty_sigma,
        )
        current = current.copy(vertices=vertices, name=f"semantic_icp_stage{stage_id}")
        log["stage"] = float(stage_id)
        logs.append(log)

    end = perf_counter()
    return WrapResult(
        initial=initial,
        wrapped=current.copy(name="semantic_icp_wrapped"),
        stage_logs=logs,
        timings={"initial_alignment": align_time - start, "semantic_icp": end - align_time},
        convergence_history=logs,
        correspondence_stats={
            "min_acceptance_ratio": float(min((x["accepted_ratio"] for x in logs), default=0.0)),
            "mean_acceptance_ratio": float(np.mean([x["accepted_ratio"] for x in logs])) if logs else 0.0,
        },
    )
