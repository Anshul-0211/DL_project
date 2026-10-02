from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def modes_for_threshold(cumulative: np.ndarray, threshold: float) -> int | None:
    """Smallest number of PCA modes whose cumulative explained variance reaches threshold."""
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0]) + 1 if len(hits) else None


def _render(out: Path, ratio: np.ndarray, cumulative: np.ndarray, thresholds: list[float], pair_count: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#ffffff"
    text = "#1f2937"
    muted = "#64748b"
    grid = "#e2e8f0"
    bar_color = "#2a78d6"
    line_color = "#eb6834"

    modes = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.bar(modes, ratio * 100.0, width=0.6, color=bar_color, label="Per-mode explained variance", zorder=2)
    ax.plot(modes, cumulative * 100.0, color=line_color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=surface, markeredgewidth=2, label="Cumulative explained variance", zorder=3)

    for t in thresholds:
        k = modes_for_threshold(cumulative, t)
        if k is None:
            continue
        ax.axhline(t * 100.0, color=muted, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
        ax.text(0.6, t * 100.0 + 0.8, f"{t * 100:.0f}% reached with {k} mode{'s' if k > 1 else ''}",
                color=text, fontsize=9, ha="left", va="bottom")

    ax.set_xticks(modes)
    ax.set_xlim(0.4, len(modes) + 0.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("PCA mode", color=text)
    ax.set_ylabel("Explained variance (%)", color=text)
    ax.set_title(f"PCA deformation prior: explained variance ({pair_count} FaMoS registrations)",
                 color=text, fontsize=12, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(loc="center right", frameon=False, labelcolor=text)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-mode and cumulative explained variance of the learned PCA deformation prior."
    )
    parser.add_argument("--summary", default="outputs/famos_deformation_prior/prior_summary.json",
                        help="prior_summary.json written by learn_famos_deformation_prior.py")
    parser.add_argument("--out", default="outputs/famos_deformation_prior/prior_explained_variance.png")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.90, 0.95],
                        help="Cumulative-variance levels to mark, as fractions in (0, 1].")
    args = parser.parse_args(argv)
    for t in args.thresholds:
        if not 0.0 < t <= 1.0:
            parser.error(f"thresholds must be in (0, 1], got {t}")

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    ratio = np.asarray(summary["explained_variance_ratio"], dtype=np.float64)
    if ratio.size == 0:
        raise SystemExit(f"No explained_variance_ratio values in {args.summary}")
    cumulative = np.cumsum(ratio)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _render(out, ratio, cumulative, args.thresholds, int(summary.get("pair_count", 0)))

    report = {
        "components": int(ratio.size),
        "total_explained_variance": float(cumulative[-1]),
        "modes_for_threshold": {f"{t:.2f}": modes_for_threshold(cumulative, t) for t in args.thresholds},
        "plot": str(out),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
