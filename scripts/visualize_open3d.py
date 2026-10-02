from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_semantic_icp.geometry import Mesh, load_mesh, topology_signature
from face_semantic_icp.template import REGION_COLORS, classify_vertices


def _open3d():
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is not installed. Install it with:\n\n"
            "  python -m pip install open3d\n\n"
            "Then rerun this script."
        ) from exc
    return o3d


def _case_paths(case_dir: Path) -> list[Path]:
    names = [
        "source_input.obj",
        "initial_flame.obj",
        "semantic_icp_wrapped.obj",
        "result_flame.obj",
    ]
    paths = [case_dir / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Case folder is missing expected mesh files:\n" + "\n".join(missing))
    return paths


def _vertex_colors(mesh: Mesh, mode: str) -> np.ndarray:
    if mode == "semantic":
        labels = mesh.semantics if mesh.semantics is not None else classify_vertices(mesh)
        return np.asarray([REGION_COLORS.get(str(label), np.array([0.72, 0.72, 0.72])) for label in labels], dtype=np.float64)
    if mode == "solid" or mesh.colors is None:
        return np.tile(np.array([[0.74, 0.62, 0.55]], dtype=np.float64), (mesh.vertex_count, 1))
    return np.clip(mesh.colors[:, :3], 0.0, 1.0)


def _prepared_mesh(mesh: Mesh, index: int, count: int, spacing: float, normalize: bool) -> Mesh:
    vertices = mesh.vertices.copy()
    if normalize:
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
        vertices = (vertices - center) / max(extent, 1e-9) * 2.0
    offset_x = (index - (count - 1) * 0.5) * spacing
    vertices[:, 0] += offset_x
    return mesh.copy(vertices=vertices)


def _to_open3d_mesh(mesh: Mesh, color_mode: str):
    o3d = _open3d()
    tri = o3d.geometry.TriangleMesh()
    tri.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    tri.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    tri.vertex_colors = o3d.utility.Vector3dVector(_vertex_colors(mesh, color_mode))
    tri.compute_vertex_normals()
    tri.compute_triangle_normals()
    return tri


def _wireframe(o3d_mesh):
    o3d = _open3d()
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
    wire.paint_uniform_color([0.02, 0.02, 0.02])
    return wire


def _print_summary(paths: list[Path]) -> None:
    print("\nMeshes loaded:")
    for path in paths:
        mesh = load_mesh(path)
        sig = topology_signature(mesh)
        print(f"  {path}")
        print(f"    vertices={sig['vertex_count']} faces={sig['face_count']} checksum={sig['face_checksum']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="True 3D Open3D viewer for Face Semantic ICP Wrap++ mesh outputs.")
    parser.add_argument("meshes", nargs="*", help="OBJ/PLY meshes to visualize.")
    parser.add_argument("--case", default=None, help="Pipeline case folder containing source_input/initial/semantic/result OBJ files.")
    parser.add_argument("--color", choices=["original", "semantic", "solid"], default="original", help="Vertex coloring mode.")
    parser.add_argument("--wire", action="store_true", help="Overlay black triangle wireframes.")
    parser.add_argument("--no-normalize", action="store_true", help="Preserve original mesh coordinates instead of fitting each mesh to a unit display size.")
    parser.add_argument("--spacing", type=float, default=2.7, help="Horizontal spacing between multiple meshes.")
    parser.add_argument("--no-frame", action="store_true", help="Hide the coordinate frame.")
    parser.add_argument("--dry-run", action="store_true", help="Load and prepare meshes, but do not open the Open3D window.")
    parser.add_argument("--window-title", default="Face Semantic ICP Wrap++ Open3D Viewer", help="Open3D window title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths: list[Path] = []
    if args.case:
        paths.extend(_case_paths(Path(args.case)))
    paths.extend(Path(path) for path in args.meshes)
    if not paths:
        default_case = Path("outputs/live_demo/mesh_case")
        if default_case.exists():
            paths = _case_paths(default_case)
        else:
            raise SystemExit("Pass mesh paths or --case outputs/live_demo/mesh_case")

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    o3d = _open3d()
    _print_summary(paths)

    geometries = []
    for index, path in enumerate(paths):
        mesh = _prepared_mesh(load_mesh(path), index, len(paths), args.spacing, normalize=not args.no_normalize)
        o3d_mesh = _to_open3d_mesh(mesh, args.color)
        geometries.append(o3d_mesh)
        if args.wire:
            geometries.append(_wireframe(o3d_mesh))

    if not args.no_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.45))

    if args.dry_run:
        print(f"\nDry run OK: prepared {len(geometries)} Open3D geometries.")
        return 0

    print("\nOpen3D controls: left drag rotate, mouse wheel zoom, right drag pan.")
    print("Close the Open3D window to return to the terminal.\n")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.window_title,
        width=1400,
        height=900,
        mesh_show_back_face=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
