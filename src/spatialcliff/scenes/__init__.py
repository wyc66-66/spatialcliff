"""Scene-set registry: build a full deterministic spatial-reasoning corpus."""
from __future__ import annotations

import json
from pathlib import Path

from .generators import FAMILY_GENERATORS, QUESTIONS

SCENE_FAMILIES = tuple(sorted(FAMILY_GENERATORS))


def build_scene_set(
    out_dir: Path,
    *,
    n_seeds: int = 20,
    difficulties: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    families: tuple[str, ...] | None = None,
    image_size: int = 640,
) -> dict:
    """Render every (family, seed, difficulty) combination to disk.

    Returns a manifest: list of dicts {id, family, seed, difficulty,
    image (relpath), answer, question}.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    families = tuple(families) if families else SCENE_FAMILIES

    manifest = []
    for fam in families:
        gen = FAMILY_GENERATORS[fam]
        for seed in range(n_seeds):
            for diff in difficulties:
                img, answer = gen(seed, diff)
                if img.width != image_size:
                    from PIL import Image

                    img = img.resize((image_size, image_size), Image.LANCZOS)
                sid = f"{fam}_s{seed:03d}_d{diff}"
                img.save(img_dir / f"{sid}.png")
                manifest.append(
                    {
                        "id": sid,
                        "family": fam,
                        "seed": seed,
                        "difficulty": diff,
                        "image": f"images/{sid}.png",
                        "answer": answer,
                        "question": QUESTIONS[fam],
                    }
                )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return {"families": list(families), "n_scenes": len(manifest)}
