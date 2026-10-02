from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .datasets import DatasetItem, read_manifest
from .demo_data import make_random_topology_face
from .geometry import Mesh, load_mesh, normalize_mesh, topology_signature, write_obj
from .semantic_icp import estimate_source_landmarks, run_semantic_icp
from .template import TemplateSpec, classify_vertices, load_template
from .texture import transfer_vertex_colors


@dataclass
class PairRecord:
    pair_id: str
    source_mesh: str
    target_mesh: str
    source_points: str
    target_vertices: str
    metadata: str
    dataset: str
    teacher: str
    source_item_id: str


def sample_source_points(mesh: Mesh, point_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vertices = mesh.vertices
    if mesh.face_count == 0 or mesh.vertex_count <= point_count:
        idx = rng.choice(mesh.vertex_count, size=point_count, replace=mesh.vertex_count < point_count)
        return vertices[idx].astype(np.float32)
    faces = mesh.faces
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    probs = areas / max(float(areas.sum()), 1e-12)
    face_ids = rng.choice(len(faces), size=point_count, p=probs)
    chosen = tri[face_ids]
    r1 = np.sqrt(rng.random(point_count))
    r2 = rng.random(point_count)
    points = (1.0 - r1)[:, None] * chosen[:, 0] + (r1 * (1.0 - r2))[:, None] * chosen[:, 1] + (r1 * r2)[:, None] * chosen[:, 2]
    return points.astype(np.float32)


def _template_target_from_source(source: Mesh, template: TemplateSpec) -> Mesh:
    source = normalize_mesh(source)
    if source.semantics is None:
        source = source.copy(semantics=classify_vertices(source))
    source_landmarks = estimate_source_landmarks(source, template.landmark_uv)
    wrapped = run_semantic_icp(template, source, source_landmarks)
    colors = transfer_vertex_colors(source, wrapped.wrapped)
    return wrapped.wrapped.copy(colors=colors, name="target_flame_teacher")


def _direct_target(source: Mesh, target_path: str, template: TemplateSpec) -> Mesh:
    target = normalize_mesh(load_mesh(target_path))
    if target.vertex_count == template.mesh.vertex_count and target.face_count == template.mesh.face_count:
        return target.copy(faces=template.mesh.faces.copy(), uvs=template.mesh.uvs, semantics=template.mesh.semantics, name="target_flame_direct")
    return _template_target_from_source(source, template)


def _write_pair(pair_dir: Path, pair_id: str, source: Mesh, target: Mesh, item: DatasetItem, teacher: str, point_count: int, seed: int) -> PairRecord:
    pair_dir.mkdir(parents=True, exist_ok=True)
    source_path = pair_dir / "source.obj"
    target_path = pair_dir / "target_flame.obj"
    points_path = pair_dir / "source_points.npz"
    target_vertices_path = pair_dir / "target_vertices.npy"
    metadata_path = pair_dir / "metadata.json"

    write_obj(source, source_path)
    write_obj(target, target_path)
    points = sample_source_points(source, point_count, seed)
    np.savez_compressed(points_path, points=points)
    np.save(target_vertices_path, target.vertices.astype(np.float32))
    metadata = {
        "pair_id": pair_id,
        "dataset_item": asdict(item),
        "teacher": teacher,
        "source_topology": topology_signature(source),
        "target_topology": topology_signature(target),
        "point_count": point_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PairRecord(
        pair_id=pair_id,
        source_mesh=str(source_path),
        target_mesh=str(target_path),
        source_points=str(points_path),
        target_vertices=str(target_vertices_path),
        metadata=str(metadata_path),
        dataset=item.dataset,
        teacher=teacher,
        source_item_id=item.item_id,
    )


def _extract_subject_id(item: "DatasetItem") -> str:
    """Extract a subject identifier from a dataset item for train/test splitting."""
    # Check metadata first
    for key in ("subject", "subject_id", "identity"):
        val = item.metadata.get(key)
        if val:
            return str(val)
    # Try to parse from source mesh path: FaMoS paths contain subject dirs like F001, M002
    import re
    src = item.source_mesh or item.item_id
    match = re.search(r"[FM]\d{3,}", str(src))
    if match:
        return match.group(0)
    # Fall back to item_id prefix (everything before first underscore)
    parts = item.item_id.split("_")
    return parts[0] if parts else item.item_id


def generate_pairs(
    manifest: str | Path | None,
    out: str | Path,
    teacher: str = "wrap++",
    template_path: str | Path | None = None,
    point_count: int = 1024,
    limit: int | None = None,
    synthetic_count: int = 0,
    seed: int = 17,
    subject_split: str | None = None,
) -> dict[str, object]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    template = load_template(template_path)
    records: list[PairRecord] = []
    rng = np.random.default_rng(seed)

    items: list[DatasetItem] = []
    if manifest:
        items.extend(item for item in read_manifest(manifest) if item.status == "ok")
    for i in range(synthetic_count):
        items.append(
            DatasetItem(
                item_id=f"synthetic_{i:06d}",
                dataset="synthetic",
                source_mesh="",
                metadata={"seed": int(seed + i), "role": "synthetic_random_topology"},
            )
        )

    # Apply subject-based train/test split
    if subject_split in ("train", "test"):
        subject_ids = sorted({_extract_subject_id(item) for item in items if item.dataset != "synthetic"})
        n_train = max(1, int(len(subject_ids) * 0.8))
        train_subjects = set(subject_ids[:n_train])
        test_subjects = set(subject_ids[n_train:])
        keep_subjects = train_subjects if subject_split == "train" else test_subjects
        items = [
            item for item in items
            if item.dataset == "synthetic" or _extract_subject_id(item) in keep_subjects
        ]

    if limit is not None:
        items = items[:limit]

    for index, item in enumerate(items):
        pair_id = f"pair_{index:06d}"
        try:
            if item.dataset == "synthetic":
                source = make_random_topology_face(seed=int(item.metadata.get("seed", seed + index)))
            else:
                source = normalize_mesh(load_mesh(item.source_mesh))
            if source.semantics is None:
                source = source.copy(semantics=classify_vertices(source))

            if item.target_mesh:
                target = _direct_target(source, item.target_mesh, template)
                used_teacher = "direct_flame_registration" if target.vertex_count == template.mesh.vertex_count else teacher
            else:
                target = _template_target_from_source(source, template)
                used_teacher = teacher
            target = target.copy(faces=template.mesh.faces.copy(), uvs=template.mesh.uvs, semantics=template.mesh.semantics)
            records.append(_write_pair(out / pair_id, pair_id, source, target, item, used_teacher, point_count, int(rng.integers(0, 2**31 - 1))))
        except Exception as exc:
            error_dir = out / pair_id
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir / "error.json").write_text(json.dumps({"pair_id": pair_id, "item": asdict(item), "error": str(exc)}, indent=2), encoding="utf-8")

    manifest_path = out / "pairs_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
    summary = {
        "out": str(out),
        "pairs_manifest": str(manifest_path),
        "pair_count": len(records),
        "teacher": teacher,
        "point_count": point_count,
        "template_topology": topology_signature(template.mesh),
        "subject_split": subject_split,
    }
    (out / "pairs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_pairs_manifest(path: str | Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PairRecord(**json.loads(line)))
    return records

