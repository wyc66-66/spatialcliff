#!/usr/bin/env python3
"""Sweep the scene-complexity grid against the scene corpus (GPU).

Configurations are the six difficulty levels (d0..d5) per family. For each
(family, difficulty) we run every seed once and record correct/total.
Output: a JSON table plus a complexity axis per family.

The complexity axis is *family-local*: what "difficulty" means depends on the
family's mechanism (object count for relpos/nearest, occluders for occlusion,
lookalike count for lookalike). We therefore emit both the raw difficulty
level and a normalized 0..1 complexity value.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from spatialcliff.check import is_correct  # noqa: E402
from spatialcliff.engine import QwenVLEngine  # noqa: E402

DIFFICULTIES = (0, 1, 2, 3, 4, 5)
# normalized complexity of each difficulty level (shared across families)
NORM = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="data/scenes", type=Path)
    ap.add_argument("--out", default="data/sweep", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--families", type=str, nargs="*", default=None)
    ap.add_argument("--difficulties", type=int, nargs="*", default=None)
    args = ap.parse_args()

    manifest = json.loads((args.scenes / "manifest.json").read_text(encoding="utf-8"))
    if args.families:
        manifest = [s for s in manifest if s["family"] in args.families]
    if args.difficulties:
        manifest = [s for s in manifest if s["difficulty"] in args.difficulties]

    engine = QwenVLEngine(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict]] = {}
    t0 = time.time()
    for fam in sorted({s["family"] for s in manifest}):
        fam_scenes = [s for s in manifest if s["family"] == fam]
        per_diff: dict[int, dict] = {}
        n_done = 0
        for s in fam_scenes:
            img = Image.open(args.scenes / s["image"]).convert("RGB")
            raw = engine.ask(img, s["question"])
            ok = is_correct(s["family"], raw, s["answer"])
            d = per_diff.setdefault(s["difficulty"], {"correct": 0, "total": 0})
            d["total"] += 1
            d["correct"] += int(ok)
            n_done += 1
        for diff, st in sorted(per_diff.items()):
            rows.setdefault(fam, []).append(
                {
                    "family": fam,
                    "difficulty": diff,
                    "label": f"d{diff}",
                    "complexity": NORM[diff],
                    "correct": st["correct"],
                    "total": st["total"],
                }
            )
        print(f"[{fam}] {n_done} scenes done in {time.time()-t0:.0f}s", flush=True)

    out = {"by_family": rows, "norm_axis": NORM, "n_scenes": len(manifest)}
    (args.out / "sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", args.out / "sweep.json")


if __name__ == "__main__":
    main()
