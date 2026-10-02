from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .geometry import load_mesh, topology_signature, vertex_normals
from .template import load_template


MESH_SUFFIXES = {".obj", ".ply"}
PRIMARY_DATASETS = {"famos", "d3dfacs", "hifi3dface", "generic"}

DATASET_CATALOG = {
    "famos": {
        "priority": "primary",
        "access": "license-gated",
        "url": "https://cove.thecvf.com/datasets/969",
        "local_hint": "D:/datasets/face_retopology/incoming/famos",
        "role": "dynamic 3D head scans with FLAME-topology registrations when licensed assets are available",
    },
    "d3dfacs": {
        "priority": "primary",
        "access": "license-gated",
        "url": "https://flame.is.tue.mpg.de/",
        "local_hint": "D:/datasets/D3DFACS_FLAME_registrations",
        "role": "FLAME/D3DFACS registered 4D face scans; strongest direct supervision target",
    },
    "hifi3dface": {
        "priority": "secondary",
        "access": "public code, large externally hosted resources",
        "url": "https://github.com/linchaobao/hifi3dface",
        "local_hint": "D:/datasets/HiFi3DFace",
        "role": "high-detail source geometry that should be wrapped into FLAME-style supervision",
    },
    "generic": {
        "priority": "utility",
        "access": "local folder",
        "url": "",
        "local_hint": "D:/datasets/generic_face_meshes",
        "role": "any local OBJ/PLY folder treated as source geometry",
    },
}


@dataclass
class DatasetItem:
    item_id: str
    dataset: str
    source_mesh: str
    target_mesh: str | None = None
    split: str = "unknown"
    subject: str | None = None
    sequence: str | None = None
    frame: str | None = None
    is_flame_topology: bool = False
    status: str = "ok"
    error: str | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _mesh_paths(root: Path, glob_pattern: str | None = None) -> Iterable[Path]:
    iterator = root.glob(glob_pattern) if glob_pattern else root.rglob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in MESH_SUFFIXES:
            yield path


def _infer_split(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    for split in ("train", "val", "valid", "validation", "test"):
        if split in parts:
            return "val" if split in {"valid", "validation"} else split
    return "unknown"


def _infer_subject_sequence(root: Path, path: Path) -> tuple[str | None, str | None, str | None]:
    rel = path.relative_to(root)
    parts = rel.parts
    subject = parts[0] if len(parts) >= 2 else None
    sequence = parts[1] if len(parts) >= 3 else None
    frame = path.stem
    return subject, sequence, frame


def _looks_like_registered(path: Path) -> bool:
    text = " ".join(p.lower() for p in path.parts)
    hints = ("flame", "registered", "registration", "template", "tracked", "tracked_mesh")
    return any(h in text for h in hints)


def _match_key(path: Path) -> str:
    key = path.stem.lower()
    for token in ("_registered", "_registration", "_flame", "_template", "_tracked", "_target", "_scan", "_source"):
        key = key.replace(token, "")
    return re_compact(key)


def re_compact(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def _target_map(target_root: Path | None, target_glob: str | None = None) -> dict[str, Path]:
    if target_root is None:
        return {}
    mapping: dict[str, Path] = {}
    for path in _mesh_paths(target_root, target_glob):
        mapping.setdefault(_match_key(path), path)
    return mapping


def _target_for_registered_dataset(dataset: str, path: Path, is_fixed: bool) -> str | None:
    if dataset in {"famos", "d3dfacs"} and (is_fixed or _looks_like_registered(path)):
        return str(path)
    return None


def dataset_catalog(local_base: str | Path = "D:/datasets") -> dict[str, object]:
    base = Path(local_base)
    entries = []
    for name, info in DATASET_CATALOG.items():
        entry_info = dict(info)
        default_hint = Path(str(entry_info.pop("local_hint")))
        local_hint = default_hint if base == Path("D:/datasets") else base / default_hint.name
        entries.append(
            {
                "name": name,
                **entry_info,
                "local_hint": str(local_hint).replace("\\", "/"),
                "local_exists": local_hint.exists(),
                "local_path": str(local_hint),
            }
        )
    return {
        "local_base": str(base),
        "entries": entries,
        "notes": [
            "No restricted dataset is auto-downloaded.",
            "FaMoS and D3DFACS/FLAME registrations require license approval before local scanning.",
            "HiFi3DFace public code/resources can be placed locally and scanned as source geometry.",
        ],
    }


def scan_dataset(
    dataset: str,
    root: str | Path,
    out: str | Path,
    template_path: str | Path | None = None,
    target_root: str | Path | None = None,
    source_glob: str | None = None,
    target_glob: str | None = None,
) -> dict[str, object]:
    dataset = dataset.lower()
    if dataset not in PRIMARY_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(sorted(PRIMARY_DATASETS))}")

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    target_root_path = Path(target_root) if target_root else None
    if target_root_path is not None and not target_root_path.exists():
        raise FileNotFoundError(f"Target root does not exist: {target_root_path}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "dataset_manifest.jsonl"
    summary_path = out / "dataset_summary.json"

    template = load_template(template_path)
    template_sig = topology_signature(template.mesh)
    known_flame_counts = {(5023, 9976), (int(template_sig["vertex_count"]), int(template_sig["face_count"]))}
    targets = _target_map(target_root_path, target_glob)

    items: list[DatasetItem] = []
    for index, mesh_path in enumerate(_mesh_paths(root, source_glob)):
        rel = mesh_path.relative_to(root).as_posix()
        subject, sequence, frame = _infer_subject_sequence(root, mesh_path)
        item = DatasetItem(
            item_id=f"{dataset}_{index:06d}",
            dataset=dataset,
            source_mesh=str(mesh_path),
            split=_infer_split(mesh_path),
            subject=subject,
            sequence=sequence,
            frame=frame,
            metadata={"relative_path": rel, "registered_name_hint": _looks_like_registered(mesh_path)},
        )
        try:
            mesh = load_mesh(mesh_path)
            _ = vertex_normals(mesh)
            item.vertex_count = mesh.vertex_count
            item.face_count = mesh.face_count
            item.is_flame_topology = (mesh.vertex_count, mesh.face_count) in known_flame_counts
            explicit_target = targets.get(_match_key(mesh_path))
            item.target_mesh = str(explicit_target) if explicit_target else _target_for_registered_dataset(dataset, mesh_path, item.is_flame_topology)
            if dataset == "hifi3dface":
                item.metadata["role"] = "source_high_detail_requires_wrap"
            elif item.target_mesh:
                item.metadata["role"] = "registered_flame_supervision"
            else:
                item.metadata["role"] = "source_requires_wrap"
        except Exception as exc:
            item.status = "error"
            item.error = str(exc)
        items.append(item)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item)) + "\n")

    ok_items = [item for item in items if item.status == "ok"]
    summary = {
        "dataset": dataset,
        "root": str(root),
        "target_root": str(target_root_path) if target_root_path else None,
        "manifest": str(manifest_path),
        "total_items": len(items),
        "ok_items": len(ok_items),
        "error_items": len(items) - len(ok_items),
        "flame_topology_items": sum(1 for item in ok_items if item.is_flame_topology),
        "items_with_target": sum(1 for item in ok_items if item.target_mesh),
        "source_glob": source_glob,
        "target_glob": target_glob,
        "template_topology": template_sig,
        "notes": _dataset_notes(dataset),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _dataset_notes(dataset: str) -> list[str]:
    if dataset == "famos":
        return ["FaMoS is license-gated and may include FLAME-topology registrations; this scanner expects local files only."]
    if dataset == "d3dfacs":
        return ["D3DFACS FLAME registrations are distributed through the FLAME ecosystem under license; this scanner expects local files only."]
    if dataset == "hifi3dface":
        return ["HiFi3DFace/HIFI3D++ meshes are treated as high-detail source geometry and wrapped into FLAME supervision."]
    return ["Generic folders are treated as source meshes requiring teacher wrapping unless a target is supplied later."]


def read_manifest(path: str | Path) -> list[DatasetItem]:
    path = Path(path)
    items: list[DatasetItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        items.append(DatasetItem(**data))
    return items
