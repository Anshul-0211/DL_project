from __future__ import annotations

import importlib.util
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, to_jsonable
from .datasets import dataset_catalog
from .template import load_template


CORE_MODULES = ["numpy", "scipy", "PIL", "cv2", "sklearn"]
OPTIONAL_MODULES = ["torch", "torchvision", "mediapipe", "insightface", "mica", "micalib", "decalib", "trimesh"]


def _module_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _gpu_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False, timeout=5)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip()}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {"available": bool(lines), "devices": lines}


def run_doctor(config: PipelineConfig | None = None) -> dict[str, Any]:
    config = config or PipelineConfig()
    template_status: dict[str, Any]
    try:
        template = load_template(
            config.template_path,
            strict_flame=config.strict_flame,
            semantic_map_path=config.semantic_map_path,
            landmark_map_path=config.landmark_map_path,
        )
        template_status = {
            "available": True,
            "official_flame": template.is_official_flame,
            "source_path": template.source_path,
            "vertex_count": template.mesh.vertex_count,
            "face_count": template.mesh.face_count,
            "warnings": template.warnings,
        }
    except Exception as exc:
        template_status = {"available": False, "error": str(exc)}

    assets = {
        "assets_dir": str(Path("assets").resolve()),
        "default_flame_template": Path("assets/flame_template.obj").exists(),
    }
    core = _module_status(CORE_MODULES)
    optional = _module_status(OPTIONAL_MODULES)
    return {
        "status": "ok" if all(core.values()) and template_status.get("available") else "attention",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "core_dependencies": core,
        "optional_adapters": optional,
        "gpu": _gpu_status(),
        "assets": assets,
        "datasets": dataset_catalog(),
        "template": template_status,
        "config": to_jsonable(config),
    }
