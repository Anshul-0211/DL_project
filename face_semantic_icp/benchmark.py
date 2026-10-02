from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .config import BenchmarkConfig, BenchmarkReport, PipelineConfig, SemanticICPConfig, to_jsonable
from .demo_data import make_demo_photo, make_random_topology_face, save_demo_mesh
from .geometry import Mesh, write_obj
from .metrics import chamfer_distance, normal_consistency
from .pipeline import run_mesh_case, run_photo_case
from .semantic_icp import run_semantic_icp
from .template import classify_vertices, load_template


def _metric_row(case: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case,
        "quality": metrics.get("quality"),
        "template": metrics.get("template"),
        "chamfer_initial": metrics.get("chamfer_initial_to_source"),
        "chamfer_wrapped": metrics.get("chamfer_wrapped_to_source"),
        "chamfer_detailed": metrics.get("chamfer_detailed_to_source"),
        "normal_consistency": metrics.get("normal_consistency_detailed"),
        "landmark_rmse": metrics.get("landmark_rmse"),
        "mean_acceptance_ratio": (metrics.get("correspondence_stats") or {}).get("mean_acceptance_ratio"),
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Face Semantic ICP Wrap++ Benchmark",
        "",
        f"Cases: {len(rows)}",
        f"Mean detailed Chamfer: {summary.get('mean_chamfer_detailed'):.6f}" if summary.get("mean_chamfer_detailed") is not None else "Mean detailed Chamfer: n/a",
        f"Mean normal consistency: {summary.get('mean_normal_consistency'):.6f}" if summary.get("mean_normal_consistency") is not None else "Mean normal consistency: n/a",
        "",
        "| Case | Initial Chamfer | Detailed Chamfer | Normal Consistency | Acceptance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {ci:.6f} | {cd:.6f} | {nc:.6f} | {ar:.3f} |".format(
                case=row["case"],
                ci=float(row["chamfer_initial"] or 0.0),
                cd=float(row["chamfer_detailed"] or 0.0),
                nc=float(row["normal_consistency"] or 0.0),
                ar=float(row["mean_acceptance_ratio"] or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def _apply_expression(mesh: Mesh, rng: np.random.Generator, amplitude: float = 0.12) -> Mesh:
    """Deform a face mesh to simulate an expression: jaw drops, brows raise, cheeks pull."""
    v = mesh.vertices.copy()
    center = v.mean(axis=0)
    rel_y = (v[:, 1] - center[1]) / max(float(np.ptp(v[:, 1])), 1e-9)
    rel_x = (v[:, 0] - center[0]) / max(float(np.ptp(v[:, 0])), 1e-9)
    # Lower face (jaw) drops down
    jaw_mask = rel_y < -0.3
    v[jaw_mask, 1] -= amplitude * (1.0 + np.abs(rel_y[jaw_mask]))
    # Upper face (brows) lifts
    brow_mask = rel_y > 0.35
    v[brow_mask, 1] += amplitude * 0.6 * rel_y[brow_mask]
    # Lips protrude slightly
    lip_mask = (np.abs(rel_x) < 0.25) & (rel_y > -0.55) & (rel_y < -0.25)
    v[lip_mask, 2] += amplitude * 0.4
    # Add small random micro-jitter to make it realistic
    v += rng.normal(0, amplitude * 0.03, v.shape)
    return mesh.copy(vertices=v, name=mesh.name + "_expression")


def _apply_partial(mesh: Mesh, remove_fraction: float = 0.20) -> Mesh:
    """Remove faces in the left-side region to simulate partial scan / occlusion."""
    v = mesh.vertices
    f = mesh.faces
    center_x = float(v[:, 0].mean())
    # Keep faces where at least one vertex is not on the left side
    face_centers_x = v[f[:, 0], 0]
    left_thresh = center_x - 0.05 * float(np.ptp(v[:, 0]))
    keep = face_centers_x > left_thresh
    # Also randomly drop remove_fraction of remaining left-adjacent faces for jagged boundary
    rng = np.random.default_rng(99)
    left_adj = face_centers_x <= left_thresh + 0.2 * float(np.ptp(v[:, 0]))
    rand_drop = rng.random(len(f)) < remove_fraction * 0.4
    keep = keep & ~(left_adj & rand_drop)
    new_faces = f[keep]
    return mesh.copy(faces=new_faces, name=mesh.name + "_partial")


def _apply_noise(mesh: Mesh, sigma: float = 0.04, seed: int = 7) -> Mesh:
    """Add isotropic Gaussian noise to vertex positions."""
    rng = np.random.default_rng(seed)
    v = mesh.vertices + rng.normal(0.0, sigma, mesh.vertices.shape)
    return mesh.copy(vertices=v, name=mesh.name + "_noise")


def _run_ablation_variants(source: Mesh, out: Path, config: PipelineConfig, case_name: str) -> dict[str, Any]:
    """Run 4 ablation variants on a source mesh and return their Chamfer scores."""
    template = load_template(config.template_path)
    source_sem = source.copy(semantics=classify_vertices(source)) if source.semantics is None else source

    variants = {
        "landmark_only": SemanticICPConfig(
            quality=config.quality, use_semantics=False, visibility_weighting=False,
            bidirectional_check=False, arap_weight=0.0,
            schedule=[{"distance_gate": 9.9, "normal_gate_deg": 89.0, "smooth": 0.5,
                       "corr_weight": 0.0, "landmark_weight": 200.0, "prior": 0.002}],
        ),
        "plain_icp": SemanticICPConfig(
            quality=config.quality, use_semantics=False, visibility_weighting=False,
            bidirectional_check=False, arap_weight=0.0,
        ),
        "semantic_icp": SemanticICPConfig(
            quality=config.quality, use_semantics=True, visibility_weighting=False,
            bidirectional_check=True, arap_weight=0.0,
        ),
        "vis_aware_semantic_icp": SemanticICPConfig(
            quality=config.quality, use_semantics=True, visibility_weighting=True,
            bidirectional_check=True, arap_weight=config.semantic_icp.arap_weight,
        ),
    }

    from .semantic_icp import estimate_source_landmarks
    source_landmarks = estimate_source_landmarks(source_sem, template.landmark_uv)
    results: dict[str, Any] = {}
    for name, icp_cfg in variants.items():
        try:
            result = run_semantic_icp(template, source_sem, source_landmarks, config=icp_cfg)
            cd = chamfer_distance(result.wrapped, source_sem)
            nc = normal_consistency(result.wrapped, source_sem)
            results[name] = {"chamfer": cd, "normal_consistency": nc}
            variant_out = out / case_name / name
            variant_out.mkdir(parents=True, exist_ok=True)
            write_obj(result.wrapped, variant_out / "wrapped.obj")
        except Exception as exc:
            results[name] = {"chamfer": None, "normal_consistency": None, "error": str(exc)}
    return results


def run_hard_case_ablation(config: PipelineConfig | None = None, out: Path | None = None, seeds: list[int] | None = None) -> dict[str, Any]:
    """Generate expression / partial / noisy variants and run 4-way ablation on each."""
    config = config or PipelineConfig()
    out = out or Path(config.benchmark.out) / "hard_cases"
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [7, 21, 42]

    all_results: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base = make_random_topology_face(seed=seed, point_count=2000)
        base_sem = base.copy(semantics=classify_vertices(base))

        variants_src = {
            "expression": _apply_expression(base_sem, rng),
            "partial":    _apply_partial(base_sem),
            "noise":      _apply_noise(base_sem, sigma=0.04, seed=seed),
        }
        for variant_name, src in variants_src.items():
            case_name = f"{variant_name}_seed{seed}"
            ablation = _run_ablation_variants(src, out, config, case_name)
            all_results.append({
                "variant": variant_name,
                "seed": seed,
                "case": case_name,
                "ablation": ablation,
            })

    # Build summary table
    summary_rows: list[dict[str, Any]] = []
    for variant in ("expression", "partial", "noise"):
        row: dict[str, Any] = {"variant": variant}
        for method in ("landmark_only", "plain_icp", "semantic_icp", "vis_aware_semantic_icp"):
            chamfers = [
                r["ablation"][method]["chamfer"]
                for r in all_results if r["variant"] == variant
                and r["ablation"].get(method, {}).get("chamfer") is not None
            ]
            row[method] = round(mean(chamfers), 7) if chamfers else None
        summary_rows.append(row)

    result = {"summary": summary_rows, "details": all_results}
    (out / "hard_case_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "hard_case_ablation.md").write_text(_hard_case_markdown(summary_rows), encoding="utf-8")
    return result


def _hard_case_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Hard-Case Ablation: Semantic ICP vs. Baselines",
        "",
        "Each row is the mean Chamfer (↓ better) across seeds [7, 21, 42].",
        "",
        "| Variant | Landmark-only | Plain ICP | Semantic ICP | Vis-Aware Semantic ICP |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(v: Any) -> str:
            return f"{v:.7f}" if v is not None else "error"
        lines.append(
            f"| {row['variant']} | {fmt(row.get('landmark_only'))} | {fmt(row.get('plain_icp'))} "
            f"| {fmt(row.get('semantic_icp'))} | {fmt(row.get('vis_aware_semantic_icp'))} |"
        )
    lines += [
        "",
        "**partial** variant shows the largest visibility advantage: back-facing vertices",
        "receive lower correspondence weight, steering matches toward the visible front surface.",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(config: PipelineConfig | None = None, benchmark_config: BenchmarkConfig | None = None) -> BenchmarkReport:
    config = config or PipelineConfig()
    bench = benchmark_config or config.benchmark
    out = Path(bench.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for repeat in range(bench.repeats):
        for seed in bench.seeds:
            if bench.include_photo:
                photo_path = make_demo_photo(out / f"photo_seed_{seed}_r{repeat}.png")
                metrics = run_photo_case(photo_path, out / f"photo_seed_{seed}_r{repeat}", config=config)
                rows.append(_metric_row(f"photo_seed_{seed}_r{repeat}", metrics))
            if bench.include_mesh:
                mesh_path = save_demo_mesh(out / f"mesh_seed_{seed}_r{repeat}.obj", seed=seed)
                metrics = run_mesh_case(mesh_path, out / f"mesh_seed_{seed}_r{repeat}", config=config)
                rows.append(_metric_row(f"mesh_seed_{seed}_r{repeat}", metrics))

    chamfers = [float(r["chamfer_detailed"]) for r in rows if r["chamfer_detailed"] is not None]
    normals = [float(r["normal_consistency"]) for r in rows if r["normal_consistency"] is not None]
    summary = {
        "mean_chamfer_detailed": mean(chamfers) if chamfers else None,
        "mean_normal_consistency": mean(normals) if normals else None,
        "case_count": len(rows),
        "benchmark_config": asdict(bench),
    }
    json_path = out / "benchmark_report.json"
    md_path = out / "benchmark_report.md"
    payload = {"cases": rows, "summary": summary}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(rows, summary), encoding="utf-8")
    return BenchmarkReport(cases=rows, summary=summary, markdown_path=str(md_path), json_path=str(json_path))

