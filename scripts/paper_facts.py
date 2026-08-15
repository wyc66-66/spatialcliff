#!/usr/bin/env python3
"""Extract the factual claims the report makes, straight from sweep.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatialcliff.analysis import SweepResult  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    args = ap.parse_args()

    result = SweepResult.load(args.sweep)

    print("== per-family summary ==")
    for fam, s in result.summary().items():
        print(f"  {fam}: best={s['best_acc']:.3f} worst={s['worst_acc']:.3f} range={s['range']:.3f} trend={s['trend']:+.3f}")
        if s["falloff"]:
            f = s["falloff"]
            print(f"    falloff at {f['at']} (acc {f['acc_before']:.3f} -> {f['acc_after']:.3f}, drop {f['drop']:.3f})")

    print("\n== complexity-sensitivity ranking (best-worst range) ==")
    rows = []
    for fam, s in result.summary().items():
        rows.append((s["range"], fam))
    for rng, fam in sorted(rows, reverse=True):
        print(f"  {fam}: {rng:.3f}")


if __name__ == "__main__":
    main()
