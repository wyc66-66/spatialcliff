#!/usr/bin/env python3
"""Render analysis figures for the SpatialCliff report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spatialcliff.analysis import SweepResult  # noqa: E402

FAMILY_LABELS = {
    "relpos": "relative position",
    "occlusion": "occlusion",
    "lookalike": "lookalike binding",
    "nearest": "nearest neighbor",
}
FAMILY_COLORS = {
    "relpos": "#58a6ff",
    "occlusion": "#f85149",
    "lookalike": "#d29922",
    "nearest": "#3fb950",
}


def fig_decay(result: SweepResult, out: Path) -> None:
    """One panel per family: accuracy vs complexity (difficulty), with CI band."""
    families = sorted(result.curves)
    n_fam = len(families)
    cols = 2
    rows = (n_fam + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.2 * rows))
    axes = axes.flatten()
    for i, fam in enumerate(families):
        ax = axes[i]
        c = result.curves[fam]
        col = FAMILY_COLORS.get(fam, "#58a6ff")
        ax.plot(c.complexity, c.acc, "-o", ms=5, lw=1.8, color=col)
        ax.fill_between(c.complexity, c.ci_lo, c.ci_hi, color=col, alpha=0.15)
        fo = c.falloff()
        if fo:
            idx = c.labels.index(fo["at"])
            ax.axvline(c.complexity[idx], color="r", ls="--", alpha=0.7)
            ax.annotate(f"falloff @ {fo['at']} ({fo['drop']*100:.0f}pt)",
                        xy=(c.complexity[idx], fo["acc_after"]),
                        xytext=(c.complexity[idx] - 0.18, fo["acc_after"] + 0.25),
                        fontsize=9, color="r",
                        arrowprops=dict(arrowstyle="->", color="r", lw=0.8))
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontsize=12)
        ax.set_xlabel("scene complexity (d0 → d5)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, color="#333")
    for j in range(n_fam, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-mechanism accuracy vs scene complexity (Qwen2.5-VL-3B)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sensitivity(result: SweepResult, out: Path) -> None:
    """Horizontal bars of simplest-minus-hardest accuracy per family."""
    summary = result.summary()
    fams = sorted(summary, key=lambda f: -summary[f]["range"])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(fams)))
    y = range(len(fams))
    ranges = [summary[f]["range"] for f in fams]
    ax.barh(list(y), ranges, color=[FAMILY_COLORS.get(f, "#58a6ff") for f in fams])
    ax.set_yticks(list(y))
    ax.set_yticklabels([FAMILY_LABELS.get(f, f) for f in fams], fontsize=10)
    ax.set_xlabel("simplest minus hardest scene accuracy (points)")
    ax.set_xlim(0, 1)
    for i, r in enumerate(ranges):
        ax.text(r + 0.01, i, f"{r:.2f}", va="center", fontsize=9)
    ax.axvline(0.15, color="r", ls=":", alpha=0.6)
    ax.text(0.16, len(fams) - 0.4, "falloff threshold (0.15)", fontsize=8, color="r")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument(
        "--figs",
        default="docs/paper/spatialcliff/figures",
        type=Path,
        help="Output directory. Files are named fig1_decay.png / fig2_sensitivity.png "
        "so the paper renderer (which inlines figures/figN_*.png) picks them up.",
    )
    args = ap.parse_args()

    result = SweepResult.load(args.sweep)
    args.figs.mkdir(parents=True, exist_ok=True)

    fig_decay(result, args.figs / "fig1_decay.png")
    fig_sensitivity(result, args.figs / "fig2_sensitivity.png")

    print(json.dumps(result.summary(), indent=1))


if __name__ == "__main__":
    main()
