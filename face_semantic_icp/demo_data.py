from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Delaunay

from .geometry import Mesh, normalize_mesh, write_obj
from .template import REGION_COLORS, classify_uv, face_surface


def make_demo_photo(path: str | Path, size: int = 768) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), (228, 232, 235))
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    face_box = [cx - 210, cy - 275, cx + 210, cy + 285]
    draw.ellipse(face_box, fill=(202, 154, 129), outline=(142, 94, 78), width=4)
    draw.ellipse([cx - 90, cy - 65, cx - 35, cy - 34], fill=(246, 247, 242), outline=(58, 45, 42), width=3)
    draw.ellipse([cx + 35, cy - 65, cx + 90, cy - 34], fill=(246, 247, 242), outline=(58, 45, 42), width=3)
    draw.ellipse([cx - 69, cy - 58, cx - 52, cy - 41], fill=(35, 46, 55))
    draw.ellipse([cx + 52, cy - 58, cx + 69, cy - 41], fill=(35, 46, 55))
    draw.arc([cx - 120, cy - 105, cx - 20, cy - 58], 190, 350, fill=(54, 37, 30), width=8)
    draw.arc([cx + 20, cy - 105, cx + 120, cy - 58], 190, 350, fill=(54, 37, 30), width=8)
    draw.polygon([(cx, cy - 52), (cx - 42, cy + 62), (cx + 34, cy + 64)], fill=(182, 121, 103), outline=(136, 84, 73))
    draw.ellipse([cx - 33, cy + 45, cx - 8, cy + 60], fill=(112, 71, 62))
    draw.ellipse([cx + 9, cy + 45, cx + 33, cy + 60], fill=(112, 71, 62))
    draw.ellipse([cx - 92, cy + 112, cx + 92, cy + 172], fill=(141, 53, 65), outline=(97, 38, 46), width=3)
    draw.rectangle([cx - 92, cy + 112, cx + 92, cy + 142], fill=(202, 154, 129))
    draw.arc([cx - 92, cy + 100, cx + 92, cy + 172], 20, 160, fill=(97, 38, 46), width=4)
    draw.arc([cx - 92, cy + 92, cx + 92, cy + 188], 200, 340, fill=(97, 38, 46), width=4)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(path)
    return path


def _skin_color_from_uv(u: np.ndarray, v: np.ndarray, labels: np.ndarray) -> np.ndarray:
    colors = np.asarray([REGION_COLORS[str(s)] for s in labels], dtype=np.float64)
    cheek = 0.05 * np.exp(-(((np.abs(u) - 0.45) / 0.20) ** 2 + ((v + 0.08) / 0.26) ** 2))
    colors[:, 0] = np.clip(colors[:, 0] + cheek, 0, 1)
    colors[:, 1] = np.clip(colors[:, 1] - 0.02 * cheek, 0, 1)
    return colors


def make_random_topology_face(seed: int = 7, point_count: int = 1850) -> Mesh:
    rng = np.random.default_rng(seed)
    pts: list[tuple[float, float]] = []
    while len(pts) < point_count:
        u = rng.uniform(-1.0, 1.0)
        v = rng.uniform(-1.0, 1.0)
        if abs(u) <= 0.97 - 0.17 * abs(v) + 0.05 * np.cos(np.pi * v):
            pts.append((u, v))

    # Feature rings make the arbitrary triangulation preserve key facial structures.
    for cx, cy, sx, sy, n in [(-0.38, 0.24, 0.19, 0.08, 42), (0.38, 0.24, 0.19, 0.08, 42), (0.0, -0.42, 0.32, 0.12, 60), (0.0, 0.02, 0.18, 0.30, 64)]:
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            pts.append((cx + sx * np.cos(a), cy + sy * np.sin(a)))

    uv = np.asarray(pts, dtype=np.float64)
    uv += rng.normal(0, 0.006, size=uv.shape)
    variant = {
        "width": 1.04,
        "jaw": 1.15,
        "nose": 1.13,
        "lips": 1.10,
        "asymmetry": 0.55,
    }
    vertices = face_surface(uv[:, 0], uv[:, 1], variant)
    detail = 0.020 * np.sin(18 * uv[:, 0] + 2.0) * np.cos(15 * uv[:, 1] - 0.5)
    vertices[:, 2] += detail
    tri = Delaunay(uv)
    faces = tri.simplices.astype(np.int64)
    labels = classify_uv(uv[:, 0], uv[:, 1])
    colors = _skin_color_from_uv(uv[:, 0], uv[:, 1], labels)
    mesh = Mesh(vertices, faces, colors=colors, uvs=np.stack([(uv[:, 0] + 1) * 0.5, (uv[:, 1] + 1) * 0.5], axis=1), semantics=labels, name="random_topology_face")
    return normalize_mesh(mesh)


def save_demo_mesh(path: str | Path, seed: int = 7) -> Path:
    path = Path(path)
    mesh = make_random_topology_face(seed=seed)
    write_obj(mesh, path)
    return path

