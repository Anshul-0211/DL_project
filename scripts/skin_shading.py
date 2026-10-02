"""Shared deterministic skin-material shader for pipeline step renders.

Used by render_showcase_steps.py so the figures share one look: smooth
per-vertex normals, a 3-light studio
rig, a warm subsurface-style shadow tint, a soft skin sheen, and gamma
correction. Still a plain matplotlib PolyCollection rasterizer -- deterministic
renders from local OBJ/PLY geometry, not AI-generated images.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection

SKIN = np.array([0.80, 0.58, 0.50])
SKIN_LIGHT = np.array([0.88, 0.68, 0.59])
SKIN_COOL = np.array([0.66, 0.70, 0.79])
TARGET = np.array([0.46, 0.71, 0.58])
PRED = np.array([0.42, 0.56, 0.82])

_SHADOW_TINT = np.array([1.06, 0.90, 0.82])

_KEY = np.array([-0.3, 0.5, 0.8])
_KEY /= np.linalg.norm(_KEY)
_FILL = np.array([0.55, 0.15, 0.5])
_FILL /= np.linalg.norm(_FILL)
_RIM = np.array([0.1, -0.35, -0.85])
_RIM /= np.linalg.norm(_RIM)
_VIEW = np.array([0.0, 0.0, 1.0])

_AMBIENT = 0.20
_KEY_I = 0.85
_FILL_I = 0.35
_RIM_I = 0.25
_GAMMA = 1.0 / 2.2


def centered(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    extent = max(float((mx - mn).max()), 1e-9)
    return (vertices - (mn + mx) * 0.5) / extent


def view_transform(vertices: np.ndarray, yaw_deg: float = -8.0, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    v = centered(vertices)
    x = v[:, 0] * cy + v[:, 2] * sy
    z = -v[:, 0] * sy + v[:, 2] * cy
    y = v[:, 1] * cp - z * sp
    z2 = v[:, 1] * sp + z * cp
    return np.stack([x, y, z2], axis=1)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (Gouraud), not per-face flat normals."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def _lit_color(base: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Linear-space 3-light Lambert + warm shadow tint + key-light sheen, then gamma."""
    n_key = np.clip(normals @ _KEY, 0.0, None)
    n_fill = np.clip(normals @ _FILL, 0.0, None)
    n_rim = np.clip(normals @ _RIM, 0.0, None) ** 1.5

    diffuse = _AMBIENT + _KEY_I * n_key + _FILL_I * n_fill + _RIM_I * n_rim
    diffuse = diffuse[:, None]

    shadow_w = (1.0 - n_key)[:, None]
    tinted_base = base[None, :] * (1.0 - 0.35 * shadow_w) + base[None, :] * _SHADOW_TINT[None, :] * (0.35 * shadow_w)

    half = _KEY + _VIEW
    half /= np.linalg.norm(half)
    spec = 0.10 * np.clip(normals @ half, 0.0, None) ** 24
    spec = spec[:, None]

    linear = np.clip(tinted_base * diffuse + spec, 0.0, 1.0)
    return np.clip(linear, 1e-6, 1.0) ** _GAMMA


def shade_faces(vertices_view: np.ndarray, faces: np.ndarray, base_color: np.ndarray) -> np.ndarray:
    """Return one RGB color per face, smooth-shaded from averaged vertex normals."""
    vn = vertex_normals(vertices_view, faces)
    face_normals = (vn[faces[:, 0]] + vn[faces[:, 1]] + vn[faces[:, 2]]) / 3.0
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-9)
    return _lit_color(base_color, face_normals)


def render_mesh(
    ax,
    mesh,
    title: str,
    subtitle: str,
    color: np.ndarray = SKIN,
    heat: np.ndarray | None = None,
    yaw: float = -8.0,
    wire: bool = False,
    card_color: str = "#ffffff",
    title_color: str = "#0d47a1",
    muted_color: str = "#5b6475",
    render_points_fn=None,
) -> None:
    if mesh.faces.ndim != 2 or mesh.faces.shape[0] == 0:
        if render_points_fn is not None:
            render_points_fn(ax, mesh.vertices, title, subtitle)
        return

    verts = view_transform(mesh.vertices, yaw_deg=yaw)
    faces = mesh.faces.astype(np.int64)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    depth = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    order = np.argsort(depth)
    fs = faces[order]
    pts = np.stack(
        [np.stack([verts[fs[:, k], 0], verts[fs[:, k], 1]], axis=-1) for k in range(3)],
        axis=1,
    ) * 0.86 + 0.5

    if heat is None:
        face_color = shade_faces(verts, faces, color)[order]
    else:
        h = np.clip(heat[fs].mean(axis=1), 0.0, 1.0)
        low = np.array([0.16, 0.45, 0.78])
        mid = np.array([0.96, 0.95, 0.88])
        high = np.array([0.88, 0.19, 0.14])
        face_color = np.where(
            h[:, None] < 0.5,
            low + (mid - low) * (h[:, None] * 2.0),
            mid + (high - mid) * ((h[:, None] - 0.5) * 2.0),
        )[order]

    edge = (0.08, 0.10, 0.13, 0.05) if wire else "none"
    coll = PolyCollection(
        pts,
        facecolors=np.c_[face_color, np.ones(len(face_color))],
        edgecolors=edge,
        linewidths=0.02 if wire else 0.0,
    )
    ax.add_collection(coll)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(card_color)
    ax.text(0.5, 1.045, title, transform=ax.transAxes, ha="center", va="bottom", color=title_color, fontsize=12.5, fontweight="bold")
    if subtitle:
        ax.text(0.5, -0.055, subtitle, transform=ax.transAxes, ha="center", va="top", color=muted_color, fontsize=8.0)
