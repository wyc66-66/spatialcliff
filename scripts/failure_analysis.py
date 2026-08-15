#!/usr/bin/env python3
"""Failure-mode analysis from the per-scene responses table.

Discovers *systematic* error patterns that a difficulty-averaged table hides:
per-corner response bias in lookalike scenes, color confusions in the other
families, and the share of failures each pattern explains. Every number it
prints is directly verifiable against data/sweep/responses.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORNERS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def corner_bias(responses: list[dict]) -> dict:
    """For lookalike: distribution of the model's reported corner, split by
    whether it was correct, plus the conditional distribution given the true
    corner."""
    la = [r for r in responses if r["family"] == "lookalike"]
    reported = Counter(r["normalized"] for r in la)
    correct_by_corner: dict[str, Counter] = defaultdict(Counter)
    for r in la:
        correct_by_corner[r["answer"]][r["normalized"]] += 1
    bottom = sum(reported[c] for c in ("bottom-left", "bottom-right"))
    n = len(la)
    return {
        "n": n,
        "reported_marginal": dict(reported),
        "bottom_share": bottom / n if n else None,
        "conditional_on_true": {k: dict(v) for k, v in correct_by_corner.items()},
    }


def confusion(responses: list[dict], family: str, top_k: int = 8) -> list[dict]:
    rows = [r for r in responses if r["family"] == family and not r["correct"] and r["normalized"]]
    counts: Counter = Counter((r["answer"], r["normalized"]) for r in rows)
    total_fails = len(rows)
    return [
        {"true": a, "predicted": b, "count": n, "share_of_failures": n / total_fails if total_fails else 0}
        for (a, b), n in counts.most_common(top_k)
    ]


def color_marginal(responses: list[dict], family: str) -> dict:
    """Marginal distribution of the model's reported color (all scenes)."""
    rows = [r for r in responses if r["family"] == family and r["normalized"]]
    cnt = Counter(r["normalized"] for r in rows)
    n = len(rows)
    return {k: v / n for k, v in sorted(cnt.items(), key=lambda x: -x[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="data/sweep/responses.json", type=Path)
    args = ap.parse_args()

    responses = json.loads(args.responses.read_text(encoding="utf-8"))

    print("== lookalike corner bias ==")
    print(json.dumps(corner_bias(responses), indent=1))

    print("\n== per-family top confusions ==")
    for fam in ("relpos", "occlusion", "nearest"):
        print(f"\n[{fam}]")
        print(json.dumps(confusion(responses, fam), indent=1))

    print("\n== reported-color marginals (all scenes, families with color answers) ==")
    for fam in ("relpos", "occlusion", "nearest"):
        print(f"  {fam}:", {k: round(v, 3) for k, v in color_marginal(responses, fam).items()})


if __name__ == "__main__":
    main()
