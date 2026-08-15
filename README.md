# SpatialCliff

**Where spatial reasoning in a VLM quietly breaks as scenes get harder.**

Multimodal models answer spatial questions — "which is left of the red cube?",
"is the blue sphere occluded behind the cylinder?" — with confidence until the
scene crosses a complexity threshold. This project stress-tests open VLM spatial
reasoning with a procedurally-generated scene corpus whose complexity is
controlled along four axes (object count, occlusion, lookalike distractors,
background clutter), and maps where accuracy falls off a cliff.

## Research question

> **Which spatial-reasoning capabilities of an open VLM survive scene
> complexity, and which collapse first?**

The four axes probe distinct mechanisms:
- *object count*: more candidates = more binding load for "which one is X?"
- *occlusion*: reasoning about partially hidden geometry
- *lookalikes*: same-shape / same-color distractors force attribute binding
- *clutter*: background noise that is irrelevant but must be ignored

Working hypothesis: **coarse relational judgments (left/right between two
prominent objects) are robust; per-object attribute binding under occlusion and
lookalike distractors degrades sharply once the scene passes a complexity
threshold.**

## Method

1. **Scene corpus.** Deterministic PIL-rendered 2D top-down scenes, 4 families
   (relative-position, occlusion, lookalike, nearest-neighbor) × 6 difficulty
   levels × 20 seeds = 480 scenes, each with a ground-truth answer.
2. **Model under test.** An open 3B-class vision-language model (Qwen2.5-VL)
   in a strict zero-shot QA protocol.
3. **Scoring.** Per-family normalizer maps free-form output to the closed answer
   set; a scene is correct iff normalized == ground truth.
4. **Cliff analysis.** Per (family, difficulty) accuracy vs complexity with
   Wilson 95% CIs; a *cliff* is a budget step where accuracy drops ≥15 points in
   one step and ≥20 points below the simplest configuration.

## Reproduce

```bash
pip install -e .[gpu,paper,ui]
python scripts/build_scenes.py --seeds 20 --out data/scenes
python scripts/run_sweep.py --scenes data/scenes --out data/sweep   # GPU
python scripts/paper_facts.py --sweep data/sweep/sweep.json
python scripts/render_figures.py --sweep data/sweep/sweep.json --figs docs/figures
python -m spatialcliff ui --port 8000
```

## Layout

```
src/spatialcliff/
  scenes/        procedural scene generation
  check.py       answer normalization
  engine.py      VLM inference wrapper
  decay.py       Wilson-CI decay + cliff detection
  ui/            FastAPI console + dashboard
scripts/         build, sweep, figures, paper facts
data/            scenes/, sweep/
docs/            figures + technical report
```
---

## Live report

The technical report, figures and every number are served at **[https://wyc66-66.github.io/spatialcliff/](https://wyc66-66.github.io/spatialcliff/)**.
