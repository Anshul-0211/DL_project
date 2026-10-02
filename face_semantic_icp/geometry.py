from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    semantics: Optional[np.ndarray] = None
    name: str = "mesh"

    def copy(self, **updates: object) -> "Mesh":
        data = {
            "vertices": self.vertices.copy(),
            "faces": self.faces.copy(),
            "colors": None if self.colors is None else self.colors.copy(),
            "uvs": None if self.uvs is None else self.uvs.copy(),
            "semantics": None if self.semantics is None else self.semantics.copy(),
            "name": self.name,
        }
        data.update(updates)
        return Mesh(**data)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def ensure_float_colors(colors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if colors is None:
        return None
    colors = np.asarray(colors, dtype=np.float64)
    if colors.max(initial=0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors[:, :3], 0.0, 1.0)


def normalize_mesh(mesh: Mesh, target_extent: float = 2.0) -> Mesh:
    """Centre and uniformly scale a mesh so its longest axis spans ``target_extent``.

    Used to put source and template meshes into a common coordinate frame
    before ICP / DeformNet inference.
    """
    mn, mx = mesh.bounds()
    center = (mn + mx) * 0.5
    extent = float(np.max(mx - mn))
    if extent <= 1e-12:
        extent = 1.0
    vertices = (mesh.vertices - center) / extent * target_extent
    return mesh.copy(vertices=vertices)


def mesh_summary(mesh: Mesh) -> dict:
    """Return a human-readable summary dict for quick inspection of a mesh.

    Includes vertex/face counts, bounding-box extents, whether optional
    attributes (colors, uvs, semantics) are present, and approximate
    surface area computed from face areas.

    Example::

        >>> summary = mesh_summary(my_mesh)
        >>> print(summary["vertex_count"], summary["approx_surface_area"])
    """
    mn, mx = mesh.bounds()
    v = mesh.vertices
    f = mesh.faces
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = float(0.5 * np.linalg.norm(cross, axis=1).sum())
    return {
        "name": mesh.name,
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "extents_xyz": (mx - mn).tolist(),
        "approx_surface_area": area,
        "has_colors": mesh.colors is not None,
        "has_uvs": mesh.uvs is not None,
        "has_semantics": mesh.semantics is not None,
    }


def face_normals(mesh: Mesh) -> np.ndarray:
    """Compute per-face unit normals using the cross-product of edge vectors."""
    v = mesh.vertices
    f = mesh.faces
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-12)


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """Compute per-vertex unit normals by averaging incident face normals."""
    normals = np.zeros_like(mesh.vertices, dtype=np.float64)
    fn = face_normals(mesh)
    for i in range(3):
        np.add.at(normals, mesh.faces[:, i], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def build_vertex_adjacency(faces: np.ndarray, vertex_count: int) -> list[list[int]]:
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return [sorted(n) for n in neighbors]


def build_cotangent_laplacian(mesh: Mesh) -> sparse.csr_matrix:
    """Cotangent-weighted Laplacian — geometrically correct vs. degree-normalized.

    For each edge (i,j), the weight is (cot α + cot β) / 2 where α, β are the
    angles opposite that edge in the two incident triangles.  Diagonal entries
    equal the sum of incident edge weights so rows sum to zero.
    """
    v, f = mesh.vertices, mesh.faces
    n = mesh.vertex_count
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(n, dtype=np.float64)
    for tri in f:
        for k in range(3):
            i, j = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
            o = int(tri[k])
            ei = v[i] - v[o]
            ej = v[j] - v[o]
            cross_len = float(np.linalg.norm(np.cross(ei, ej)))
            if cross_len < 1e-12:
                continue
            cot = float(np.dot(ei, ej)) / cross_len
            w = 0.5 * cot
            rows += [i, j]
            cols += [j, i]
            data += [-w, -w]
            diag[i] += w
            diag[j] += w
    rows += list(range(n))
    cols += list(range(n))
    data += diag.tolist()
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def topology_signature(mesh: Mesh) -> dict[str, object]:
    return {
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "face_checksum": int(np.abs(mesh.faces.astype(np.int64)).sum()),
        "first_faces": mesh.faces[:8].astype(int).tolist(),
    }


def load_mesh(path: str | Path) -> Mesh:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix == ".ply":
        return load_ply(path)
    raise ValueError(f"Unsupported mesh format: {path.suffix}")


def load_obj(path: str | Path) -> Mesh:
    path = Path(path)
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_vt: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
                else:
                    colors.append([0.72, 0.58, 0.50])
            elif parts[0] == "vt" and len(parts) >= 3:
                texcoords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f" and len(parts) >= 4:
                vids: list[int] = []
                vtids: list[int] = []
                for token in parts[1:]:
                    chunks = token.split("/")
                    vi = int(chunks[0])
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    vids.append(vi - 1)
                    if len(chunks) > 1 and chunks[1]:
                        vti = int(chunks[1])
                        if vti < 0:
                            vti = len(texcoords) + vti + 1
                        vtids.append(vti - 1)
                for i in range(1, len(vids) - 1):
                    faces.append([vids[0], vids[i], vids[i + 1]])
                    if len(vtids) == len(vids):
                        face_vt.append([vtids[0], vtids[i], vtids[i + 1]])

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    color_arr = ensure_float_colors(np.asarray(colors, dtype=np.float64)) if colors else None
    uvs = None
    if texcoords and face_vt:
        uvs = np.zeros((len(vertices), 2), dtype=np.float64)
        counts = np.zeros((len(vertices), 1), dtype=np.float64)
        vt = np.asarray(texcoords, dtype=np.float64)
        for tri, uvtri in zip(faces, face_vt):
            for vi, ti in zip(tri, uvtri):
                uvs[vi] += vt[ti, :2]
                counts[vi] += 1.0
        known = counts[:, 0] > 0
        uvs[known] /= counts[known]
    return Mesh(v, f, color_arr, uvs, name=path.stem)


def load_ply(path: str | Path) -> Mesh:
    path = Path(path)
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"PLY header not terminated: {path}")
    header_end += len(b"end_header")
    while header_end < len(raw) and raw[header_end:header_end+1] in (b"\r", b"\n"):
        header_end += 1
    header = raw[:header_end].decode("ascii", errors="ignore")

    fmt = "ascii"
    vertex_count = 0
    face_count = 0
    vert_props: list[tuple[str, str]] = []   # (type, name)
    in_vert = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1] if len(parts) > 1 else "ascii"
        elif parts[0] == "element":
            in_vert = len(parts) > 1 and parts[1] == "vertex"
            if len(parts) > 2:
                if parts[1] == "vertex":
                    vertex_count = int(parts[2])
                elif parts[1] == "face":
                    face_count = int(parts[2])
        elif parts[0] == "property" and in_vert and len(parts) >= 3:
            vert_props.append((parts[1], parts[-1]))

    prop_names = [p[1] for p in vert_props]
    prop_types = [p[0] for p in vert_props]
    has_color = all(c in prop_names for c in ("red", "green", "blue"))

    if fmt == "ascii":
        return _load_ply_ascii(path, raw[header_end:].decode("ascii", errors="ignore"),
                               vertex_count, face_count, prop_names, has_color)
    else:
        return _load_ply_binary(path, raw, header_end, vertex_count, face_count,
                                prop_names, prop_types, has_color,
                                big_endian=(fmt == "binary_big_endian"))


def _load_ply_ascii(path: Path, body: str, vertex_count: int, face_count: int,
                    prop_names: list[str], has_color: bool) -> Mesh:
    lines = [l for l in body.splitlines() if l.strip()]
    vertex_lines = lines[:vertex_count]
    face_lines = lines[vertex_count: vertex_count + face_count]
    prop_idx = {p: i for i, p in enumerate(prop_names)}
    vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    colors = np.zeros((vertex_count, 3), dtype=np.float64)
    for i, line in enumerate(vertex_lines):
        vals = line.split()
        vertices[i] = [float(vals[prop_idx[a]]) for a in ("x", "y", "z")]
        if has_color:
            colors[i] = [float(vals[prop_idx[a]]) for a in ("red", "green", "blue")]
    faces: list[list[int]] = []
    for line in face_lines:
        vals = [int(x) for x in line.split()]
        ids = vals[1: 1 + vals[0]]
        for k in range(1, len(ids) - 1):
            faces.append([ids[0], ids[k], ids[k + 1]])
    return Mesh(vertices, np.asarray(faces, dtype=np.int64),
                ensure_float_colors(colors) if has_color else None, name=path.stem)


_PLY_STRUCT = {
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
    "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
    "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
    "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
}


def _load_ply_binary(path: Path, raw: bytes, data_offset: int,
                     vertex_count: int, face_count: int,
                     prop_names: list[str], prop_types: list[str],
                     has_color: bool, big_endian: bool) -> Mesh:
    end = ">" if big_endian else "<"
    stride = sum(_PLY_STRUCT[t][1] for t in prop_types)
    fmt_str = end + "".join(_PLY_STRUCT[t][0] for t in prop_types)
    prop_idx = {p: i for i, p in enumerate(prop_names)}

    vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    colors = np.zeros((vertex_count, 3), dtype=np.float64)
    offset = data_offset
    for i in range(vertex_count):
        vals = struct.unpack_from(fmt_str, raw, offset)
        offset += stride
        vertices[i] = [vals[prop_idx[a]] for a in ("x", "y", "z")]
        if has_color:
            colors[i] = [vals[prop_idx[a]] / 255.0 if prop_types[prop_idx[a]] in ("uchar", "uint8") else vals[prop_idx[a]]
                         for a in ("red", "green", "blue")]

    faces: list[list[int]] = []
    for _ in range(face_count):
        count_type = "B" if (offset < len(raw) and raw[offset] < 16) else "B"
        n_verts = struct.unpack_from(end + "B", raw, offset)[0]
        offset += 1
        idx_fmt = end + "I" * n_verts
        ids = list(struct.unpack_from(idx_fmt, raw, offset))
        offset += 4 * n_verts
        for k in range(1, len(ids) - 1):
            faces.append([ids[0], ids[k], ids[k + 1]])

    return Mesh(vertices, np.asarray(faces, dtype=np.int64),
                ensure_float_colors(colors) if has_color else None, name=path.stem)


def write_obj(mesh: Mesh, path: str | Path, texture_name: Optional[str] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = path.with_suffix(".mtl").name
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Face Semantic ICP Wrap export\n")
        if texture_name:
            handle.write(f"mtllib {mtl_name}\nusemtl face_texture\n")
        colors = mesh.colors
        for i, p in enumerate(mesh.vertices):
            if colors is not None:
                c = np.clip(colors[i], 0.0, 1.0)
                handle.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                handle.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        if mesh.uvs is not None:
            for uv in mesh.uvs:
                handle.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for tri in mesh.faces:
            if mesh.uvs is not None:
                handle.write("f " + " ".join(f"{int(i) + 1}/{int(i) + 1}" for i in tri) + "\n")
            else:
                handle.write("f " + " ".join(str(int(i) + 1) for i in tri) + "\n")
    if texture_name:
        with path.with_suffix(".mtl").open("w", encoding="utf-8") as handle:
            handle.write("newmtl face_texture\n")
            handle.write("Kd 1.0 1.0 1.0\n")
            handle.write(f"map_Kd {texture_name}\n")


def write_ply(mesh: Mesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = mesh.colors
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {mesh.vertex_count}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {mesh.face_count}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(mesh.vertices):
            if colors is not None:
                c = np.clip(colors[i] * 255.0, 0, 255).astype(int)
                handle.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}\n")
            else:
                handle.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for tri in mesh.faces:
            handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

