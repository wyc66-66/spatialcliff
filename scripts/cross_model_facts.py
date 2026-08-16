#!/usr/bin/env python3
"""Cross-model audit-transfer facts: Qwen2.5-VL-3B vs InternVL2.5-2B.

The single-model limitation of the audited sweep is addressed by re-running
the identical protocol on a second, architecturally distinct VLM (InternViT +
InternLM2 vs Qwen2.5-VL). This script re-derives the report's cross-model
claims from the two sweep JSONs:

- per-family per-difficulty accuracy for both models
- complexity trend (Pearson r) and falloff detection per family
- the lookalike bottom-quadrant bias (77% for Qwen) recomputed for InternVL
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

sys_path_guard = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="data/sweep/sweep.json", type=Path)
    ap.add_argument("--internvl", default="data/sweep_internvl/sweep.json", type=Path)
    args = ap.parse_args()

    q = json.loads(args.qwen.read_text(encoding="utf-8"))
    i = json.loads(args.internvl.read_text(encoding="utf-8"))

    print("== per-family per-difficulty accuracy (Qwen2.5-VL-3B vs InternVL2.5-2B) ==")
    table = {}
    for fam in q["by_family"]:
        q_rows = {r["difficulty"]: r for r in q["by_family"][fam]}
        i_rows = {r["difficulty"]: r for r in i["by_family"][fam]}
        fam_table = []
        for d in sorted(q_rows):
            qr = q_rows[d]["correct"] / q_rows[d]["total"]
            ir = i_rows[d]["correct"] / i_rows[d]["total"]
            fam_table.append(
                {"difficulty": d, "qwen": qr, "internvl": ir}
            )
            print(f"  {fam:<10} d{d}: qwen={qr:.2f} internvl={ir:.2f}")
        table[fam] = fam_table

    print("\n== trends (Pearson r over difficulty) ==")
    trends = {}
    for fam in q["by_family"]:
        rq = _trend(q["by_family"][fam])
        ri = _trend(i["by_family"][fam])
        trends[fam] = {"qwen": rq, "internvl": ri}
        print(f"  {fam:<10} qwen={rq:+.2f} internvl={ri:+.2f}")

    print("\n== falloffs ==")
    falloffs = {}
    for fam in q["by_family"]:
        fq = _falloff(q["by_family"][fam])
        fi = _falloff(i["by_family"][fam])
        falloffs[fam] = {"qwen": fq, "internvl": fi}
        print(f"  {fam:<10} qwen={fq} internvl={fi}")

    print("\n== lookalike bottom-quadrant bias ==")
    q_resp = json.loads(Path(str(args.qwen).replace("sweep.json", "responses.json")).read_text(encoding="utf-8"))
    i_resp = json.loads(Path(str(args.internvl).replace("sweep.json", "responses.json")).read_text(encoding="utf-8"))
    bias = {
        "qwen": _bottom_bias(q_resp),
        "internvl": _bottom_bias(i_resp),
    }
    print(f"  qwen:     {bias['qwen']}")
    print(f"  internvl: {bias['internvl']}")

    out = {
        "per_family_per_diff": table,
        "trends": trends,
        "falloffs": falloffs,
        "lookalike_bottom_bias": bias,
        "n_scenes": {"qwen": q["n_scenes"], "internvl": i["n_scenes"]},
    }
    out_path = args.qwen.parent / "cross_model.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


def _trend(rows: list[dict]) -> float:
    xs = np.array([r["complexity"] for r in rows], dtype=float)
    ys = np.array([r["correct"] / r["total"] for r in rows], dtype=float)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return 0.0
    r = np.corrcoef(xs, ys)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def _falloff(rows: list[dict]) -> dict | None:
    vals = [r["correct"] / r["total"] for r in sorted(rows, key=lambda r: r["difficulty"])]
    base = vals[0]
    prev = base
    for i, a in enumerate(vals[1:], start=1):
        if prev - a >= 0.15 and base - a >= 0.20:
            return {"at": f"d{i}", "acc_before": round(prev, 3), "acc_after": round(a, 3),
                    "drop": round(prev - a, 3)}
        prev = a
    return None


def _bottom_bias(responses: list[dict]) -> dict:
    lk = [r for r in responses if r["family"] == "lookalike"]
    n_bottom = sum(1 for r in lk if r["normalized"] in ("bottom-left", "bottom-right"))
    # share conditional on each true corner
    corners = ("top-left", "top-right", "bottom-left", "bottom-right")
    per_corner = {}
    for tc in corners:
        rows = [r for r in lk if r["answer"] == tc]
        bh = sum(1 for r in rows if r["normalized"] in ("bottom-left", "bottom-right"))
        per_corner[tc] = {"n": len(rows), "bottom_half": bh, "share": bh / len(rows) if rows else 0.0}
    return {"n": len(lk), "bottom_half": n_bottom, "share": n_bottom / len(lk) if lk else 0.0,
            "per_corner": per_corner}


if __name__ == "__main__":
    main()
