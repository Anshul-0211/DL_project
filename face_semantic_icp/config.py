from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class LandmarkConfig:
    detector: str = "auto"
    min_face_size: int = 64
    prefer_3d: bool = True
    use_mediapipe: bool = True
    use_insightface: bool = True
    use_mica_deca_imports: bool = True


@dataclass
class SemanticICPConfig:
    quality: str = "balanced"
    use_semantics: bool = True
    visibility_weighting: bool = True
    bidirectional_check: bool = True
    point_to_plane_ratio: float = 0.55
    min_acceptance_ratio: float = 0.12
    outlier_percentile: float = 98.0
    feature_anchor_weight: float = 1.0
    arap_weight: float = 0.08
    pca_weight: float = 0.03
    pca_prior_path: str | None = None
    schedule: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TextureTransferConfig:
    texture_size: int = 512
    detail_strength: float = 0.45
    max_detail_distance: float = 0.10
    symmetric_photo_fill: bool = True
    save_detail_residual: bool = True


@dataclass
class BenchmarkConfig:
    out: str = "outputs/benchmark"
    repeats: int = 1
    seeds: list[int] = field(default_factory=lambda: [7, 21, 42])
    include_photo: bool = True
    include_mesh: bool = True


@dataclass
class PipelineConfig:
    quality: str = "balanced"
    strict_flame: bool = False
    save_ablation: bool = False
    template_path: str | None = None
    semantic_map_path: str | None = None
    landmark_map_path: str | None = None
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    semantic_icp: SemanticICPConfig = field(default_factory=SemanticICPConfig)
    texture: TextureTransferConfig = field(default_factory=TextureTransferConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    deformnet_checkpoint: str | None = None


@dataclass
class LandmarkResult:
    points_2d: list[list[float]]
    points_3d: list[list[float]]
    confidences: list[float]
    source: str
    bbox: tuple[int, int, int, int] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PipelineArtifacts:
    out_dir: str
    source_input: str
    initial_flame: str
    semantic_icp_wrapped: str
    result_obj: str
    result_ply: str
    texture: str
    metrics: str
    viewer: str
    manifest: str
    semantic_overlay: str | None = None
    detail_residual: str | None = None
    ablation_report: str | None = None


@dataclass
class BenchmarkReport:
    cases: list[dict[str, Any]]
    summary: dict[str, Any]
    markdown_path: str
    json_path: str


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        return data
    known = {f.name: f for f in fields(instance)}
    values = {f.name: getattr(instance, f.name) for f in fields(instance)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            values[key] = _merge_dataclass(current, value)
        else:
            values[key] = value
    return type(instance)(**values)


def load_pipeline_config(path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
    config = PipelineConfig()
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = _merge_dataclass(config, data)
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    if clean_overrides:
        config = _merge_dataclass(config, clean_overrides)
    config.semantic_icp.quality = config.quality
    return config


def quality_schedule(quality: str) -> list[dict[str, float]]:
    presets = {
        "fast": [
            {"distance_gate": 0.40, "normal_gate_deg": 82.0, "smooth": 11.0, "corr_weight": 1.2, "landmark_weight": 65.0, "prior": 0.020},
            {"distance_gate": 0.24, "normal_gate_deg": 68.0, "smooth": 4.8, "corr_weight": 2.4, "landmark_weight": 95.0, "prior": 0.016},
        ],
        "balanced": [
            {"distance_gate": 0.42, "normal_gate_deg": 78.0, "smooth": 18.0, "corr_weight": 0.9, "landmark_weight": 55.0, "prior": 0.020},
            {"distance_gate": 0.30, "normal_gate_deg": 66.0, "smooth": 8.0, "corr_weight": 1.8, "landmark_weight": 85.0, "prior": 0.018},
            {"distance_gate": 0.20, "normal_gate_deg": 55.0, "smooth": 3.2, "corr_weight": 3.0, "landmark_weight": 120.0, "prior": 0.015},
        ],
        "best": [
            {"distance_gate": 0.46, "normal_gate_deg": 82.0, "smooth": 24.0, "corr_weight": 0.8, "landmark_weight": 60.0, "prior": 0.024},
            {"distance_gate": 0.34, "normal_gate_deg": 72.0, "smooth": 12.0, "corr_weight": 1.6, "landmark_weight": 95.0, "prior": 0.020},
            {"distance_gate": 0.24, "normal_gate_deg": 61.0, "smooth": 5.0, "corr_weight": 2.8, "landmark_weight": 130.0, "prior": 0.016},
            {"distance_gate": 0.16, "normal_gate_deg": 50.0, "smooth": 2.0, "corr_weight": 4.2, "landmark_weight": 150.0, "prior": 0.012},
        ],
    }
    return presets.get(quality, presets["balanced"])
