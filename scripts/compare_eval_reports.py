from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = {
    # metric key in eval_report rows -> True if lower is better
    "learned_chamfer": True,
    "learned_normal_consistency": False,
}


def load_rows(path: Path) -> dict[str, dict]:
    """Map pair_id -> row from an eval_report.json written by evaluate_deformnet."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = report.get("rows") or []
    if not rows:
        raise ValueError(f"No per-pair rows in {path}")
    return {row["pair_id"]: row for row in rows}


def paired_comparison(base: dict[str, dict], cand: dict[str, dict], metric: str, lower_is_better: bool) -> dict:
    """Compare two models pair-by-pair on the pairs both reports share (and both have a value for).

    A paired comparison answers "on how many of the same faces does the candidate win?",
    which a difference of means alone can hide (one outlier can move a mean).
    """
    shared = sorted(
        pid for pid in base.keys() & cand.keys()
        if base[pid].get(metric) is not None and cand[pid].get(metric) is not None
    )
    if not shared:
        raise ValueError(f"No shared pairs with '{metric}' values")
    b = np.asarray([base[p][metric] for p in shared], dtype=np.float64)
    c = np.asarray([cand[p][metric] for p in shared], dtype=np.float64)
    gain = (b - c) if lower_is_better else (c - b)  # > 0 means the candidate is better
    rel = gain / np.where(np.abs(b) > 0, np.abs(b), np.nan)
    worst = int(np.argmin(gain))
    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "pairs": len(shared),
        "baseline_mean": float(b.mean()),
        "candidate_mean": float(c.mean()),
        "mean_relative_change": float((c.mean() - b.mean()) / b.mean()) if b.mean() != 0 else None,
        "median_paired_relative_gain": float(np.nanmedian(rel)),
        "candidate_wins": int((gain > 0).sum()),
        "ties": int((gain == 0).sum()),
        "baseline_wins": int((gain < 0).sum()),
        "largest_regression_pair": shared[worst] if gain[worst] < 0 else None,
    }


def to_markdown(summary: dict, base_label: str, cand_label: str) -> str:
    lines = [
        f"# Paired eval comparison: {cand_label} vs {base_label}",
        "",
        f"| Metric | {base_label} mean | {cand_label} mean | Change of means | Median paired gain | {cand_label} wins | {base_label} wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary["metrics"]:
        arrow = "(lower better)" if s["lower_is_better"] else "(higher better)"  # ASCII: safe on cp1252 consoles
        change = "-" if s["mean_relative_change"] is None else f"{s['mean_relative_change'] * 100:+.1f}%"
        lines.append(
            f"| {s['metric']} {arrow} | {s['baseline_mean']:.6f} | {s['candidate_mean']:.6f} | {change} | "
            f"{s['median_paired_relative_gain'] * 100:+.1f}% | {s['candidate_wins']}/{s['pairs']} | {s['baseline_wins']}/{s['pairs']} |"
        )
    lines += ["", "Positive paired gain = candidate better on that pair (direction-aware per metric).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair-by-pair comparison of two DeformNet eval_report.json files evaluated on the same pairs "
                    "(default: committed PointNet v2 vs GNN v3 showcase evals)."
    )
    parser.add_argument("--baseline", default="showcase_2026_07_04/step_06_eval/v2/eval_report.json")
    parser.add_argument("--candidate", default="showcase_2026_07_04/step_06_eval/v3/eval_report.json")
    parser.add_argument("--baseline-label", default="PointNet v2")
    parser.add_argument("--candidate-label", default="GNN v3")
    parser.add_argument("--out", default=None, help="Optional directory to write comparison.json / comparison.md.")
    args = parser.parse_args(argv)

    base = load_rows(Path(args.baseline))
    cand = load_rows(Path(args.candidate))
    summary = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "unmatched_pairs": len(base.keys() ^ cand.keys()),
        "metrics": [paired_comparison(base, cand, m, lower) for m, lower in METRICS.items()],
    }
    markdown = to_markdown(summary, args.baseline_label, args.candidate_label)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
