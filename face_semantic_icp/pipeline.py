from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from .config import PipelineArtifacts, PipelineConfig, load_pipeline_config, to_jsonable
from .demo_data import make_demo_photo, make_random_topology_face, save_demo_mesh
from .geometry import Mesh, load_mesh, normalize_mesh, topology_signature, write_obj, write_ply
from .landmarks import detect_layered_landmarks, landmark_result_points_3d, read_image
from .metrics import build_metrics
from .semantic_icp import estimate_source_landmarks, run_semantic_icp
from .template import TemplateSpec, classify_vertices, load_template
from .texture import bake_texture, sample_image_colors, transfer_normal_detail_with_config, transfer_vertex_colors
from .viewer import write_viewer


def _prepare_source(mesh: Mesh) -> Mesh:
    mesh = normalize_mesh(mesh)
    semantics = mesh.semantics if mesh.semantics is not None else classify_vertices(mesh)
    return mesh.copy(semantics=semantics)


def _photo_proxy_mesh(image: Image.Image, bbox: tuple[int, int, int, int]) -> Mesh:
    seed = int(image.width * 31 + image.height * 17 + bbox[2] * 13 + bbox[3])
    source = make_random_topology_face(seed=seed, point_count=1700)
    if source.uvs is not None:
        colors = sample_image_colors(image, source.uvs, bbox)
        source = source.copy(colors=colors)
    return source.copy(name="photo_proxy_random_topology")


def _make_config(config: PipelineConfig | None = None, template_path: str | Path | None = None, quality: str | None = None, save_ablation: bool | None = None, strict_flame: bool | None = None) -> PipelineConfig:
    if config is None:
        config = load_pipeline_config()
    if template_path is not None:
        config.template_path = str(template_path)
    if quality is not None:
        config.quality = quality
        config.semantic_icp.quality = quality
    if save_ablation is not None:
        config.save_ablation = save_ablation
    if strict_flame is not None:
        config.strict_flame = strict_flame
    return config


def _load_configured_template(config: PipelineConfig) -> TemplateSpec:
    return load_template(
        config.template_path,
        strict_flame=config.strict_flame,
        semantic_map_path=config.semantic_map_path,
        landmark_map_path=config.landmark_map_path,
    )


def _write_semantic_overlay(mesh: Mesh, path: Path, size: int = 512) -> Path:
    from .template import REGION_COLORS

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), (242, 244, 247))
    draw = ImageDraw.Draw(img)
    if mesh.uvs is None or mesh.semantics is None:
        img.save(path)
        return path
    radius = max(1, size // 160)
    for uv, label in zip(mesh.uvs, mesh.semantics):
        x = int(np.clip(uv[0], 0, 1) * (size - 1))
        y = int((1.0 - np.clip(uv[1], 0, 1)) * (size - 1))
        color = tuple(np.clip(REGION_COLORS.get(str(label), np.array([0.7, 0.52, 0.45])) * 255, 0, 255).astype(int).tolist())
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    img.save(path)
    return path


def _write_case_outputs(out_dir: Path, source: Mesh, template: TemplateSpec, wrapped: Mesh, detailed: Mesh, initial: Mesh, metrics: dict[str, object], config: PipelineConfig, photo_path: Path | None = None, residual: np.ndarray | None = None, ablation_report: dict[str, object] | None = None) -> PipelineArtifacts:
    source_path = out_dir / "source_input.obj"
    initial_path = out_dir / "initial_flame.obj"
    wrapped_path = out_dir / "semantic_icp_wrapped.obj"
    texture_path = out_dir / "texture.png"
    result_obj_path = out_dir / "result_flame.obj"
    result_ply_path = out_dir / "result_flame.ply"
    metrics_path = out_dir / "metrics.json"
    viewer_path = out_dir / "viewer.html"
    manifest_path = out_dir / "artifact_manifest.json"
    overlay_path = out_dir / "semantic_overlay.png"
    residual_path = out_dir / "detail_residual.npy"
    ablation_path = out_dir / "ablation_report.json"

    write_obj(source, source_path)
    write_obj(initial, initial_path)
    write_obj(wrapped, wrapped_path)
    bake_texture(detailed, texture_path, size=config.texture.texture_size, symmetric_fill=config.texture.symmetric_photo_fill)
    write_obj(detailed, result_obj_path, texture_name=texture_path.name)
    write_ply(detailed, result_ply_path)
    _write_semantic_overlay(detailed, overlay_path, size=config.texture.texture_size)
    if residual is not None and config.texture.save_detail_residual:
        np.save(residual_path, residual)
    else:
        residual_path = None
    if ablation_report is not None:
        ablation_path.write_text(json.dumps(ablation_report, indent=2), encoding="utf-8")
    else:
        ablation_path = None
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_viewer(out_dir / "viewer.html", "Visibility-Aware Face Semantic ICP Wrap", [source, initial, wrapped, detailed], metrics, photo_path, ablation_report=ablation_report)
    signature = topology_signature(template.mesh)
    result_signature = topology_signature(detailed)
    if signature["vertex_count"] != result_signature["vertex_count"] or signature["face_checksum"] != result_signature["face_checksum"]:
        raise AssertionError("Template topology changed during export")
    artifacts = PipelineArtifacts(
        out_dir=str(out_dir),
        source_input=str(source_path),
        initial_flame=str(initial_path),
        semantic_icp_wrapped=str(wrapped_path),
        result_obj=str(result_obj_path),
        result_ply=str(result_ply_path),
        texture=str(texture_path),
        metrics=str(metrics_path),
        viewer=str(viewer_path),
        manifest=str(manifest_path),
        semantic_overlay=str(overlay_path),
        detail_residual=str(residual_path) if residual_path else None,
        ablation_report=str(ablation_path) if ablation_path else None,
    )
    manifest = {
        "method": "Visibility-Aware Face Semantic ICP Wrap",
        "artifacts": to_jsonable(artifacts),
        "template": {
            "official_flame": template.is_official_flame,
            "source_path": template.source_path,
            "warnings": template.warnings,
            "topology": topology_signature(template.mesh),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifacts


def _deformnet_warm_start(config: PipelineConfig, source: Mesh, template: TemplateSpec) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """Return (initial_vertices, uncertainty_sigma) from DeformNet if checkpoint configured."""
    ckpt_path = getattr(config, "deformnet_checkpoint", None)
    if not ckpt_path:
        return None, None
    try:
        import torch
        from .train import load_checkpoint
        from .pairs import sample_source_points
        model, ckpt_tpl, _ = load_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        if ckpt_tpl.vertex_count != template.mesh.vertex_count:
            return None, None
        pts = sample_source_points(source, 2048, seed=0)
        pts_t = torch.tensor(pts[None].astype(np.float32))
        tpl_t = torch.tensor(template.mesh.vertices.astype(np.float32))
        with torch.no_grad():
            pred, sigma = model.predict_with_uncertainty(pts_t, tpl_t)
        return pred[0].cpu().numpy(), sigma[0].cpu().numpy()
    except Exception:
        return None, None


def _finish_case(source: Mesh, out: Path, template: TemplateSpec, source_landmarks: np.ndarray, landmark_confidences: np.ndarray | None, config: PipelineConfig, photo_path: Path | None = None, landmark_source: str | None = None) -> dict[str, object]:
    detail_start = perf_counter()
    initial_vertices, uncertainty_sigma = _deformnet_warm_start(config, source, template)
    wrap = run_semantic_icp(template, source, source_landmarks, config.semantic_icp, landmark_confidences, initial_vertices=initial_vertices, uncertainty_sigma=uncertainty_sigma)
    detailed, residual = transfer_normal_detail_with_config(source, wrap.wrapped, config.texture)
    if photo_path is not None and detailed.uvs is not None:
        image = read_image(photo_path)
        bbox = None
        if metrics_bbox := getattr(config, "_photo_bbox", None):
            bbox = metrics_bbox
        if bbox is None:
            bbox = (0, 0, image.width, image.height)
        colors = sample_image_colors(image, detailed.uvs, bbox)
    else:
        colors = transfer_vertex_colors(source, detailed)
    detailed = detailed.copy(colors=colors, name="result_flame_detail_texture")
    detail_end = perf_counter()
    ablation_report = run_ablation_report(template, source, source_landmarks, config, landmark_confidences) if config.save_ablation else None
    timings = {**wrap.timings, "detail_texture": detail_end - detail_start - wrap.timings["initial_alignment"] - wrap.timings["semantic_icp"]}
    metrics = build_metrics(source, wrap.initial, wrap.wrapped, detailed, template.landmark_indices, source_landmarks, timings)
    metrics["method"] = "Visibility-Aware Face Semantic ICP Wrap"
    if template.is_official_flame:
        metrics["template"] = "official_flame_obj"
    elif template.source_path:
        metrics["template"] = "custom_template_obj"
    else:
        metrics["template"] = "FLAME_DEMO_fallback"
    metrics["template_warnings"] = template.warnings
    metrics["stage_logs"] = wrap.stage_logs
    metrics["convergence_history"] = wrap.convergence_history
    metrics["correspondence_stats"] = wrap.correspondence_stats
    metrics["quality"] = config.quality
    metrics["landmark_source"] = landmark_source
    artifacts = _write_case_outputs(out, source, template, wrap.wrapped, detailed, wrap.initial, metrics, config, photo_path, residual, ablation_report)
    metrics["artifacts"] = to_jsonable(artifacts)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_ablation_report(template: TemplateSpec, source: Mesh, source_landmarks: np.ndarray, config: PipelineConfig, landmark_confidences: np.ndarray | None = None) -> dict[str, object]:
    variants = {
        "landmark_only": {"use_semantics": False, "visibility_weighting": False, "schedule": []},
        "nonsemantic_icp": {"use_semantics": False, "visibility_weighting": False},
        "semantic_icp": {"use_semantics": True, "visibility_weighting": False},
        "visibility_aware_semantic_icp": {"use_semantics": True, "visibility_weighting": True},
    }
    rows: list[dict[str, object]] = []
    from dataclasses import replace
    from .metrics import chamfer_distance
    from .semantic_icp import initial_alignment

    for name, overrides in variants.items():
        if name == "landmark_only":
            initial = initial_alignment(template, source, source_landmarks)
            rows.append(
                {
                    "variant": name,
                    "chamfer_wrapped": chamfer_distance(initial, source),
                    "chamfer_detailed": chamfer_distance(initial, source),
                    "mean_acceptance_ratio": 0.0,
                    "stages": [],
                }
            )
            continue
        icp_config = replace(config.semantic_icp)
        for key, value in overrides.items():
            setattr(icp_config, key, value)
        wrapped = run_semantic_icp(template, source, source_landmarks, icp_config, landmark_confidences)
        detailed, _ = transfer_normal_detail_with_config(source, wrapped.wrapped, config.texture)
        rows.append(
            {
                "variant": name,
                "chamfer_wrapped": chamfer_distance(wrapped.wrapped, source),
                "chamfer_detailed": chamfer_distance(detailed, source),
                "mean_acceptance_ratio": wrapped.correspondence_stats.get("mean_acceptance_ratio", 0.0) if wrapped.correspondence_stats else 0.0,
                "stages": wrapped.stage_logs,
            }
        )
    return {"method": "Visibility-Aware Face Semantic ICP Wrap ablation", "variants": rows}


def run_mesh_case(mesh_path: str | Path, out_dir: str | Path, template_path: str | Path | None = None, config: PipelineConfig | None = None, quality: str | None = None, save_ablation: bool | None = None, strict_flame: bool | None = None) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    config = _make_config(config, template_path, quality, save_ablation, strict_flame)
    template = _load_configured_template(config)
    source = _prepare_source(load_mesh(mesh_path))
    source_landmarks = estimate_source_landmarks(source, template.landmark_uv)
    t1 = perf_counter()
    metrics = _finish_case(source, out, template, source_landmarks, None, config, landmark_source="mesh_uv_nearest")
    metrics["timings_sec"]["load_prepare"] = t1 - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_photo_case(photo_path: str | Path, out_dir: str | Path, template_path: str | Path | None = None, config: PipelineConfig | None = None, quality: str | None = None, save_ablation: bool | None = None, strict_flame: bool | None = None) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    config = _make_config(config, template_path, quality, save_ablation, strict_flame)
    template = _load_configured_template(config)
    image = read_image(photo_path)
    landmark_result = detect_layered_landmarks(image, config.landmark)
    if landmark_result.bbox is None:
        raise ValueError("Photo landmark detection did not return a face bounding box")
    bbox = landmark_result.bbox
    landmark_targets = normalize_mesh(Mesh(landmark_result_points_3d(landmark_result), np.zeros((0, 3), dtype=np.int64))).vertices
    landmark_confidences = np.asarray(landmark_result.confidences, dtype=np.float64)
    source = _photo_proxy_mesh(image, bbox)
    config._photo_bbox = landmark_result.bbox
    t1 = perf_counter()
    metrics = _finish_case(source, out, template, landmark_targets, landmark_confidences, config, Path(photo_path), landmark_source=landmark_result.source)
    metrics["landmark_notes"] = landmark_result.notes
    metrics["timings_sec"]["photo_prepare"] = t1 - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_demo(out_dir: str | Path, template_path: str | Path | None = None, config: PipelineConfig | None = None, quality: str | None = None, save_ablation: bool | None = None, strict_flame: bool | None = None) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = _make_config(config, template_path, quality, save_ablation, strict_flame)
    photo_path = make_demo_photo(out / "input_photo.png")
    mesh_path = save_demo_mesh(out / "random_topology_source.obj")
    photo_metrics = run_photo_case(photo_path, out / "photo_case", config=config)
    mesh_metrics = run_mesh_case(mesh_path, out / "mesh_case", config=config)
    index = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Face Semantic ICP Wrap Demo</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f6f7f9;color:#1d232b}}a{{display:block;margin:12px 0;font-size:18px}}</style>
</head><body>
<h1>Face Semantic ICP Wrap Demo</h1>
<a href="photo_case/viewer.html">Photo to fixed topology viewer</a>
<a href="mesh_case/viewer.html">Random-topology mesh to fixed topology viewer</a>
<p>Outputs include OBJ, PLY, texture PNG, and metrics JSON in each case directory.</p>
</body></html>"""
    (out / "index.html").write_text(index, encoding="utf-8")
    return {"photo_case": photo_metrics, "mesh_case": mesh_metrics, "index": str(out / "index.html")}
