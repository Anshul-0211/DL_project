from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .config import TextureTransferConfig
from .geometry import Mesh, vertex_normals
from .template import REGION_COLORS


def sample_image_colors(image: Image.Image, uvs: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    x, y, w, h = bbox
    px = np.clip(x + uvs[:, 0] * w, 0, image.width - 1)
    py = np.clip(y + (1.0 - uvs[:, 1]) * h, 0, image.height - 1)
    ix = np.round(px).astype(int)
    iy = np.round(py).astype(int)
    return arr[iy, ix, :3]


def transfer_vertex_colors(source: Mesh, target: Mesh) -> np.ndarray:
    if source.colors is None:
        if target.semantics is not None:
            return np.asarray([REGION_COLORS.get(str(s), np.array([0.70, 0.52, 0.45])) for s in target.semantics], dtype=np.float64)
        return np.full((target.vertex_count, 3), [0.70, 0.52, 0.45], dtype=np.float64)
    tree = cKDTree(source.vertices)
    _, idx = tree.query(target.vertices, k=1)
    return np.clip(source.colors[idx], 0.0, 1.0)


def compute_normal_detail_residual(source: Mesh, wrapped: Mesh, max_distance: float = 0.10) -> np.ndarray:
    tree = cKDTree(source.vertices)
    dist, idx = tree.query(wrapped.vertices, k=1)
    normals = vertex_normals(wrapped)
    delta = source.vertices[idx] - wrapped.vertices
    normal_disp = np.sum(delta * normals, axis=1)
    gate = np.clip(1.0 - dist / max_distance, 0.0, 1.0)
    lo, hi = np.percentile(normal_disp, [2, 98])
    normal_disp = np.clip(normal_disp, lo, hi)
    return normal_disp * gate


def transfer_normal_detail(source: Mesh, wrapped: Mesh, max_distance: float = 0.10, strength: float = 0.45) -> Mesh:
    normals = vertex_normals(wrapped)
    residual = compute_normal_detail_residual(source, wrapped, max_distance)
    vertices = wrapped.vertices + normals * (residual * strength)[:, None]
    return wrapped.copy(vertices=vertices)


def transfer_normal_detail_with_config(source: Mesh, wrapped: Mesh, config: TextureTransferConfig) -> tuple[Mesh, np.ndarray]:
    normals = vertex_normals(wrapped)
    residual = compute_normal_detail_residual(source, wrapped, config.max_detail_distance)
    vertices = wrapped.vertices + normals * (residual * config.detail_strength)[:, None]
    return wrapped.copy(vertices=vertices), residual


def _mirror_fill_texture(tex: np.ndarray) -> np.ndarray:
    flipped = tex[:, ::-1, :]
    mask = tex.mean(axis=2, keepdims=True) < 0.02
    return np.where(mask, flipped, tex)


def bake_texture(mesh: Mesh, path: str | Path, size: int = 512, symmetric_fill: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    uvs = mesh.uvs
    colors = mesh.colors
    if uvs is None:
        img = Image.new("RGB", (size, size), (178, 132, 113))
        img.save(path)
        return path
    if colors is None:
        colors = np.full((mesh.vertex_count, 3), [0.70, 0.52, 0.45], dtype=np.float64)

    px = np.clip(np.round(uvs[:, 0] * (size - 1)).astype(int), 0, size - 1)
    py = np.clip(np.round((1.0 - uvs[:, 1]) * (size - 1)).astype(int), 0, size - 1)
    known = np.stack([px, py], axis=1)
    tree = cKDTree(known.astype(np.float64))
    grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    _, idx = tree.query(coords, k=1)
    tex = colors[idx].reshape(size, size, 3)
    if symmetric_fill:
        tex = _mirror_fill_texture(tex)
    tex = np.clip(tex * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(tex, mode="RGB").save(path)
    return path
