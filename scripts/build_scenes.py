#!/usr/bin/env python3
"""Build the spatial-reasoning scene corpus (CPU only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatialcliff.scenes import build_scene_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="data/scenes", type=Path)
    args = ap.parse_args()

    meta = build_scene_set(args.out, n_seeds=args.seeds)
    print(json.dumps(meta))
    print("manifest:", args.out / "manifest.json")


if __name__ == "__main__":
    main()
