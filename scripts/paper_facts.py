#!/usr/bin/env python3
"""Extract the factual claims the report makes, straight from sweep.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatialcliff.analysis import SweepResult, paired_mcnemar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--responses", default="data/sweep/responses.json", type=Path)
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

    print("\n== paired McNemar relpos d0 vs d1 (same seed) ==")
    resp = json.loads(args.responses.read_text(encoding="utf-8"))
    rows = [r for r in resp if r["family"] == "relpos" and r["difficulty"] in (0, 1)]
    by_seed: dict[int, dict[int, bool]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["difficulty"]] = r["correct"]
    pairs = [(v[0], v[1]) for v in by_seed.values() if 0 in v and 1 in v]
    m = paired_mcnemar([x for x, _ in pairs], [y for _, y in pairs])
    print(f"  paired seeds = {len(pairs)}")
    print(f"  b (d0 ok -> d1 fail) = {m['b']}, c (d0 fail -> d1 ok) = {m['c']}, "
          f"discordant = {m['n_discordant']}")
    print(f"  exact McNemar two-sided p = {m['p']:.4f}")


if __name__ == "__main__":
    main()
