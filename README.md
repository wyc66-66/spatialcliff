# SpatialCliff

**Does scene complexity really break VLM spatial reasoning — or does the benchmark's answer-color shortcut fake the cliff?**

Multimodal models answer spatial questions — "which is left of the red cube?",
"is the blue sphere occluded behind the cylinder?" — with confidence until the
scene crosses a complexity threshold. This project stress-tests open VLM spatial
reasoning with a procedurally-generated scene corpus whose complexity is
controlled along four axes (object count, occlusion, lookalike distractors,
background clutter) — and then **audits the task design itself**.

The audit found and removed a color-identity shortcut in the first sweep
(§2.1): the three color-bearing families used a fixed answer color, so a model
could answer correctly by reporting the singleton color without doing any
spatial reasoning. After randomizing answer colors and adding same-colored
distractors, the dramatic "cliffs" of the first sweep largely disappeared —
replaced by gradual decay (trend r as low as −0.86 for occlusion) and one
genuine mechanism boundary (`lookalike` binding) with a strong bottom-quadrant
attribution bias (77% of reported corners are bottom-half).

## Research question

> **Which spatial-reasoning capabilities of an open VLM survive scene
> complexity — and how much of an observed "cliff" is a task-design artifact?**

The four axes probe distinct mechanisms:
- *object count*: more candidates = more binding load for "which one is X?"
- *occlusion*: reasoning about partially hidden geometry
- *lookalikes*: same-shape / same-color distractors force attribute binding
- *clutter*: background noise that is irrelevant but must be ignored

## Method

1. **Scene corpus.** Deterministic PIL-rendered 2D top-down scenes, 4 families
   (relative-position, occlusion, lookalike, nearest-neighbor) × 6 difficulty
   levels × 40 seeds = 960 scenes, each with a ground-truth answer.
2. **Shortcut audit.** In the color-bearing families the answer color is
   sampled per scene and ≥1 same-colored distractor object is placed; a model
   that reports "the rare color" now scores at chance.
3. **Model under test.** An open 3B-class vision-language model (Qwen2.5-VL)
   in a strict zero-shot QA protocol; every raw response is saved for audit.
4. **Analysis.** Per (family, difficulty) accuracy with Wilson 95% CIs; a
   *cliff* is a difficulty step where accuracy drops ≥15 points in one step
   and ≥20 points below the simplest configuration. A *trend* r (Pearson) also
   captures gradual decay that cliffs miss.

## Reproduce

```bash
pip install -e .[gpu,paper,ui]
python scripts/build_scenes.py --seeds 40 --out data/scenes
python scripts/run_sweep.py --scenes data/scenes --out data/sweep   # GPU
python scripts/paper_facts.py --sweep data/sweep/sweep.json
python scripts/failure_analysis.py --responses data/sweep/responses.json
python scripts/render_figures.py --sweep data/sweep/sweep.json --figs docs/figures
python scripts/render_spatialcliff_paper.py
python -m spatialcliff ui --port 8000
```

The pre-audit (fixed answer-color) sweep is archived at
`data/sweep_pre_audit/sweep.json`; the report discusses both in appendix A.

## Layout

```
src/spatialcliff/
  scenes/        procedural scene generation (shortcut-free answer colors)
  check.py       answer normalization
  engine.py      VLM inference wrapper
  analysis.py    Wilson-CI decay + cliff detection + trend
  ui/            FastAPI console + dashboard
scripts/         build, sweep, figures, failure analysis, paper facts
data/            scenes/, sweep/, responses/, sweep_pre_audit/
docs/            figures + technical report
```
---

## Live report

The technical report, figures and every number are served at **[https://wyc66-66.github.io/spatialcliff/](https://wyc66-66.github.io/spatialcliff/)**.
