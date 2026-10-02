from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .geometry import Mesh, load_mesh, write_obj
from .metrics import chamfer_distance, normal_consistency
from .pairs import read_pairs_manifest
from .semantic_icp import estimate_source_landmarks, run_semantic_icp
from .template import classify_vertices, load_template
from .train import _device, _torch, load_checkpoint
from .viewer import write_viewer


def _markdown(rows: list[dict[str, object]], subject_split: str | None = None) -> str:
    header = "# DeformNet Evaluation"
    if subject_split:
        header += f" — {subject_split} split (cross-subject)"
    def fmt(value: object, digits: int) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

    lines = [
        header,
        "",
        "| Pair | Learned Chamfer | ICP Chamfer | Learned Normal | ICP Normal | Runtime Learned | Runtime ICP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {pair_id} | {lc} | {ic} | {ln} | {inn} | {lr} | {ir} |".format(
                pair_id=row["pair_id"],
                lc=fmt(row.get("learned_chamfer"), 6),
                ic=fmt(row.get("icp_chamfer"), 6),
                ln=fmt(row.get("learned_normal_consistency"), 6),
                inn=fmt(row.get("icp_normal_consistency"), 6),
                lr=fmt(row.get("learned_runtime_sec"), 4),
                ir=fmt(row.get("icp_runtime_sec"), 4),
            )
        )
    return "\n".join(lines) + "\n"


def evaluate_deformnet(
    checkpoint: str | Path,
    pairs: str | Path,
    out: str | Path,
    template_path: str | Path | None = None,
    baseline: str = "wrap++",
    device: str = "cuda",
    limit: int | None = None,
    subject_split: str | None = None,
) -> dict[str, object]:
    torch, _, _, _ = _torch()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device_obj = _device(device)
    model, checkpoint_template, _ = load_checkpoint(checkpoint, map_location=str(device_obj))
    model.to(device_obj)
    model.eval()
    template_vertices = torch.tensor(checkpoint_template.vertices.astype(np.float32), device=device_obj)
    template = load_template(template_path)
    records = read_pairs_manifest(pairs)
    if limit is not None:
        records = records[:limit]
    rows: list[dict[str, object]] = []

    for index, record in enumerate(records):
        source = load_mesh(record.source_mesh)
        if source.semantics is None:
            source = source.copy(semantics=classify_vertices(source))
        target = load_mesh(record.target_mesh).copy(name="target_flame")
        points = torch.tensor(np.load(record.source_points)["points"][None].astype(np.float32), device=device_obj)

        t0 = perf_counter()
        with torch.no_grad():
            pred_vertices = model(points, template_vertices)[0].detach().cpu().numpy()
        learned_runtime = perf_counter() - t0
        pred = Mesh(pred_vertices, checkpoint_template.faces.copy(), name="learned_deformnet")
        row = {
            "pair_id": record.pair_id,
            "learned_chamfer": chamfer_distance(pred, target),
            "learned_normal_consistency": normal_consistency(target, pred),
            "learned_runtime_sec": learned_runtime,
        }

        if baseline == "wrap++":
            b0 = perf_counter()
            source_landmarks = estimate_source_landmarks(source, template.landmark_uv)
            wrap = run_semantic_icp(template, source, source_landmarks)
            baseline_mesh = wrap.wrapped.copy(name="wrappp_baseline")
            row.update(
                {
                    "icp_chamfer": chamfer_distance(baseline_mesh, target),
                    "icp_normal_consistency": normal_consistency(target, baseline_mesh),
                    "icp_runtime_sec": perf_counter() - b0,
                }
            )
        else:
            baseline_mesh = template.mesh.copy(name="template_baseline")
            row.update({"icp_chamfer": None, "icp_normal_consistency": None, "icp_runtime_sec": None})

        if index == 0:
            write_obj(pred, out / "sample_eval_prediction.obj")
            write_viewer(out / "sample_eval_viewer.html", "DeformNet Evaluation", [source, baseline_mesh, pred, target], row)
        rows.append(row)

    learned = [float(row["learned_chamfer"]) for row in rows]
    icp = [float(row["icp_chamfer"]) for row in rows if row.get("icp_chamfer") is not None]
    report = {
        "checkpoint": str(checkpoint),
        "pairs": str(pairs),
        "baseline": baseline,
        "device": str(device_obj),
        "pair_count": len(rows),
        "mean_learned_chamfer": float(np.mean(learned)) if learned else None,
        "mean_icp_chamfer": float(np.mean(icp)) if icp else None,
        "subject_split": subject_split,
        "rows": rows,
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "eval_report.md").write_text(_markdown(rows, subject_split=subject_split), encoding="utf-8")
    return report
