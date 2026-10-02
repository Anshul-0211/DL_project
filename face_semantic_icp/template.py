from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Mesh, load_mesh, normalize_mesh


REGION_NAMES = np.array(["forehead", "brows", "eyes", "nose", "lips", "cheeks", "jaw", "ears_neck"])

REGION_COLORS = {
    "forehead": np.array([0.76, 0.61, 0.52]),
    "brows": np.array([0.28, 0.20, 0.17]),
    "eyes": np.array([0.83, 0.86, 0.86]),
    "nose": np.array([0.82, 0.62, 0.52]),
    "lips": np.array([0.63, 0.25, 0.30]),
    "cheeks": np.array([0.78, 0.56, 0.47]),
    "jaw": np.array([0.68, 0.49, 0.42]),
    "ears_neck": np.array([0.60, 0.45, 0.39]),
}


@dataclass
class TemplateSpec:
    mesh: Mesh
    landmark_indices: np.ndarray
    landmark_uv: np.ndarray
    is_official_flame: bool = False
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)


def face_surface(u: np.ndarray, v: np.ndarray, variant: dict[str, float] | None = None) -> np.ndarray:
    variant = variant or {}
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    width_scale = variant.get("width", 1.0)
    jaw_scale = variant.get("jaw", 1.0)
    nose_scale = variant.get("nose", 1.0)
    lip_scale = variant.get("lips", 1.0)
    asym = variant.get("asymmetry", 0.0)

    width = (0.16 + 0.58 * np.sqrt(np.clip(1.0 - (0.80 * v) ** 2, 0.0, 1.0))) * width_scale
    width *= 1.0 + 0.10 * jaw_scale * np.exp(-((v + 0.62) / 0.28) ** 2)
    x = u * width
    y = 1.13 * v

    dome = 0.17 * np.sqrt(np.clip(1.0 - (0.72 * u) ** 2 - (0.48 * v) ** 2, 0.0, 1.0))
    nose = 0.43 * nose_scale * np.exp(-((u / 0.15) ** 2 + ((v - 0.05) / 0.27) ** 2))
    nose += 0.12 * nose_scale * np.exp(-((u / 0.11) ** 2 + ((v - 0.28) / 0.30) ** 2))
    lips = 0.13 * lip_scale * np.exp(-((u / 0.30) ** 2 + ((v + 0.41) / 0.09) ** 2))
    lip_groove = -0.055 * np.exp(-((u / 0.26) ** 2 + ((v + 0.405) / 0.022) ** 2))
    eyes = -0.08 * (
        np.exp(-(((u - 0.38) / 0.18) ** 2 + ((v - 0.25) / 0.09) ** 2))
        + np.exp(-(((u + 0.38) / 0.18) ** 2 + ((v - 0.25) / 0.09) ** 2))
    )
    brows = 0.045 * (
        np.exp(-(((u - 0.38) / 0.24) ** 2 + ((v - 0.42) / 0.08) ** 2))
        + np.exp(-(((u + 0.38) / 0.24) ** 2 + ((v - 0.42) / 0.08) ** 2))
    )
    chin = 0.055 * np.exp(-((u / 0.32) ** 2 + ((v + 0.82) / 0.16) ** 2))
    z = dome + nose + lips + lip_groove + eyes + brows + chin
    x = x + asym * 0.035 * (1.0 - v**2)
    return np.stack([x, y, z], axis=-1)


def classify_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u)
    v = np.asarray(v)
    labels = np.full(u.shape, "cheeks", dtype=object)
    labels[v > 0.55] = "forehead"
    labels[(v > 0.34) & (v <= 0.55) & (np.abs(u) < 0.62)] = "brows"
    labels[(np.abs(u - 0.38) < 0.22) & (np.abs(v - 0.24) < 0.12)] = "eyes"
    labels[(np.abs(u + 0.38) < 0.22) & (np.abs(v - 0.24) < 0.12)] = "eyes"
    labels[(np.abs(u) < 0.20) & (v > -0.20) & (v < 0.35)] = "nose"
    labels[(np.abs(u) < 0.34) & (v > -0.53) & (v < -0.29)] = "lips"
    labels[v < -0.62] = "jaw"
    labels[np.abs(u) > 0.88] = "ears_neck"
    return labels.astype(str)


def infer_uv_from_vertices(vertices: np.ndarray) -> np.ndarray:
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_norm = x / max(np.max(np.abs(x)), 1e-9)
    y_norm = y / max(np.max(np.abs(y)), 1e-9)
    return np.stack([(x_norm + 1.0) * 0.5, (y_norm + 1.0) * 0.5], axis=1)


def canonical_landmark_uv() -> np.ndarray:
    pts: list[tuple[float, float]] = []

    # Jaw, 17 points.
    for t in np.linspace(-1.0, 1.0, 17):
        y = -0.81 + 0.18 * abs(t) ** 1.7
        pts.append((0.82 * t, y))

    # Brows, 10 points.
    for side in (-1, 1):
        for t in np.linspace(0.0, 1.0, 5):
            x = side * (0.55 - 0.30 * t)
            y = 0.43 + 0.035 * np.sin(np.pi * t)
            pts.append((x, y))

    # Nose bridge and base, 9 points.
    for y in np.linspace(0.28, -0.16, 4):
        pts.append((0.0, y))
    for x in np.linspace(-0.20, 0.20, 5):
        pts.append((x, -0.22 + 0.04 * (1.0 - abs(x) / 0.20)))

    # Eyes, 12 points.
    for cx in (-0.38, 0.38):
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            pts.append((cx + 0.17 * np.cos(a), 0.24 + 0.075 * np.sin(a)))

    # Outer mouth, 12 points.
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        pts.append((0.30 * np.cos(a), -0.42 + 0.12 * np.sin(a)))

    # Inner mouth, 8 points.
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        pts.append((0.18 * np.cos(a), -0.42 + 0.052 * np.sin(a)))

    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] != 68:
        raise AssertionError(f"Expected 68 landmarks, got {arr.shape[0]}")
    return arr


def nearest_indices_for_uv(mesh: Mesh, landmark_uv: np.ndarray) -> np.ndarray:
    if mesh.uvs is not None:
        uv_norm = mesh.uvs.copy()
        target = np.stack([(landmark_uv[:, 0] + 1.0) * 0.5, (landmark_uv[:, 1] + 1.0) * 0.5], axis=1)
        distances = ((uv_norm[None, :, :] - target[:, None, :]) ** 2).sum(axis=2)
    else:
        target_xyz = face_surface(landmark_uv[:, 0], landmark_uv[:, 1])
        target_xyz = target_xyz / max(np.max(np.abs(target_xyz)), 1e-9)
        v = mesh.vertices / max(np.max(np.abs(mesh.vertices)), 1e-9)
        distances = ((v[None, :, :] - target_xyz[:, None, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def create_demo_template(nx: int = 57, ny: int = 73, variant: dict[str, float] | None = None, name: str = "FLAME_DEMO") -> TemplateSpec:
    us = np.linspace(-1.0, 1.0, nx)
    vs = np.linspace(-1.0, 1.0, ny)
    uu, vv = np.meshgrid(us, vs)
    vertices = face_surface(uu.ravel(), vv.ravel(), variant)
    uvs = np.stack([(uu.ravel() + 1.0) * 0.5, (vv.ravel() + 1.0) * 0.5], axis=1)
    faces: list[list[int]] = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    semantics = classify_uv(uu.ravel(), vv.ravel())
    colors = np.asarray([REGION_COLORS[s] for s in semantics], dtype=np.float64)
    mesh = Mesh(vertices, np.asarray(faces, dtype=np.int64), colors=colors, uvs=uvs, semantics=semantics, name=name)
    landmark_uv = canonical_landmark_uv()
    landmark_indices = nearest_indices_for_uv(mesh, landmark_uv)
    return TemplateSpec(mesh=normalize_mesh(mesh), landmark_indices=landmark_indices, landmark_uv=landmark_uv, is_official_flame=False)


def classify_vertices(mesh: Mesh) -> np.ndarray:
    uv = mesh.uvs if mesh.uvs is not None else infer_uv_from_vertices(mesh.vertices)
    u = uv[:, 0] * 2.0 - 1.0
    v = uv[:, 1] * 2.0 - 1.0
    return classify_uv(u, v)


def _load_semantic_map(path: str | Path | None, vertex_count: int) -> np.ndarray | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = np.full(vertex_count, "cheeks", dtype=object)
    if isinstance(data, list):
        if len(data) != vertex_count:
            raise ValueError(f"Semantic map has {len(data)} labels but template has {vertex_count} vertices")
        return np.asarray(data, dtype=str)
    if not isinstance(data, dict):
        raise ValueError("Semantic map must be a list of labels or a dict of region to vertex indices")
    for region, indices in data.items():
        labels[np.asarray(indices, dtype=np.int64)] = str(region)
    return labels.astype(str)


def _load_landmark_map(path: str | Path | None, mesh: Mesh, landmark_uv: np.ndarray) -> np.ndarray:
    if path is None:
        return nearest_indices_for_uv(mesh, landmark_uv)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ordered = [data[str(i)] for i in range(len(landmark_uv))]
    elif isinstance(data, list):
        ordered = data
    else:
        raise ValueError("Landmark map must be a list of vertex indices or dict keyed by landmark id")
    if len(ordered) < len(landmark_uv):
        raise ValueError(f"Landmark map has {len(ordered)} entries but {len(landmark_uv)} are required")
    indices = np.asarray(ordered[: len(landmark_uv)], dtype=np.int64)
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= mesh.vertex_count:
        raise ValueError("Landmark map contains vertex index outside template range")
    return indices


def validate_template(mesh: Mesh, strict_flame: bool = False) -> list[str]:
    warnings: list[str] = []
    if mesh.vertex_count < 1000 or mesh.face_count < 1000:
        warnings.append("Template is very small for face retopology; use only for tests or demos.")
    if mesh.faces.min(initial=0) < 0 or mesh.faces.max(initial=0) >= mesh.vertex_count:
        raise ValueError("Template faces reference vertices outside the vertex array")
    if strict_flame and (mesh.vertex_count != 5023 or mesh.face_count != 9976):
        raise ValueError("Strict FLAME mode expects a 5023-vertex / 9976-face FLAME topology OBJ")
    if mesh.vertex_count != 5023 or mesh.face_count != 9976:
        warnings.append("Template is not the common FLAME 5023/9976 topology; topology will still be preserved.")
    return warnings


def load_template(
    path: str | Path | None = None,
    strict_flame: bool = False,
    semantic_map_path: str | Path | None = None,
    landmark_map_path: str | Path | None = None,
) -> TemplateSpec:
    explicit = Path(path) if path else None
    default = Path("assets/flame_template.obj")
    if explicit and not explicit.exists():
        if strict_flame:
            raise FileNotFoundError(f"Template path does not exist: {explicit}")
        template_path = None
    else:
        template_path = explicit if explicit and explicit.exists() else default if default.exists() else None
    if strict_flame and template_path is None:
        raise FileNotFoundError("Strict FLAME mode requires --template or assets/flame_template.obj")
    if template_path is None:
        return create_demo_template()

    mesh = normalize_mesh(load_mesh(template_path))
    warnings = validate_template(mesh, strict_flame=strict_flame)
    if mesh.uvs is None:
        mesh = mesh.copy(uvs=infer_uv_from_vertices(mesh.vertices))
    semantics = _load_semantic_map(semantic_map_path, mesh.vertex_count)
    if semantics is None:
        semantics = classify_vertices(mesh)
    if mesh.colors is None:
        mesh = mesh.copy(colors=np.asarray([REGION_COLORS[s] for s in semantics], dtype=np.float64))
    is_official_flame = mesh.vertex_count == 5023 and mesh.face_count == 9976
    mesh = mesh.copy(semantics=semantics, name="FLAME_TEMPLATE" if is_official_flame else template_path.stem)
    landmark_uv = canonical_landmark_uv()
    landmark_indices = _load_landmark_map(landmark_map_path, mesh, landmark_uv)
    return TemplateSpec(mesh=mesh, landmark_indices=landmark_indices, landmark_uv=landmark_uv, is_official_flame=is_official_flame, source_path=str(template_path), warnings=warnings)
