# SpatialCliff

**Does scene complexity really break VLM spatial reasoning — or does the benchmark's answer-color shortcut fake the cliff?**

Multimodal models answer spatial questions — "which is left of the red cube?",
"is the blue sphere occluded behind the cylinder?" — with confidence until the
scene crosses a complexity threshold. This project stress-tests open VLM spatial
reasoning with a procedurally-generated scene corpus: four spatial-reasoning
mechanisms (relative position, occlusion, lookalike binding, nearest neighbor),
each stepped through six complexity levels — and then **audits the task design
itself**.

The methodology follows the fine-grained-diagnostic-benchmark philosophy of
TemporalBench: controlled stimuli, per-mechanism measurement, and a hard look at
what the model actually does when it fails. TemporalBench showed that aggregated
accuracy hides fine-grained temporal failures (GPT-4o reaches only 38.5% on
fine-grained temporal QA), and its MBA correction showed that a benchmark's own
task design can manufacture a failure. This project applies the same philosophy
to *spatial* reasoning — and the same self-audit.

The audit found and removed a color-identity shortcut in the first sweep
(§2.1): the three color-bearing families used a fixed answer color, so a model
could answer correctly by reporting the singleton color without doing any
spatial reasoning. After randomizing answer colors and adding same-colored
distractors, the dramatic "cliffs" of the first sweep largely disappeared —
replaced by gradual decay (trend r as low as −0.86 for occlusion) and one
genuine mechanism boundary (`lookalike` binding) with a strong bottom-quadrant
attribution bias (77% of reported corners are bottom-half).

**Scale:** 4 mechanisms × 6 complexity levels × 40 seeds = **960 scenes**, each
with a ground-truth answer computed from the layout itself; every raw model
response is archived for audit (`data/sweep/responses.json`).

## Research question

> **Which spatial-reasoning capabilities of an open VLM survive scene
> complexity — and how much of an observed "cliff" is a task-design artifact?**

The four mechanisms probe distinct capabilities:
- *relative position*: binding a property of a spatially-related object, under
  increasing candidate counts (3→15 objects)
- *occlusion*: recovering a hidden property through an occluder, with more
  occluders and objects (occluders 1→5, objects 3→11)
- *lookalikes*: binding a small attribute to the right instance among identical
  distractors (2→9 identical circles)
- *nearest neighbor*: relative-distance computation over a growing scene (3→14
  objects)

Within each mechanism, a *harder scene* jointly raises object count, shrinks
object scale, and adds background clutter (Table 2.1 of the report); the
measured curves are the joint effect of scene density on that mechanism.

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
python scripts/render_figures.py --sweep data/sweep/sweep.json   # -> docs/paper/spatialcliff/figures/
python scripts/render_spatialcliff_paper.py
python -m spatialcliff ui --port 8000
```

The pre-audit (fixed answer-color) sweep is archived at
`data/sweep_pre_audit/sweep.json`; the report discusses both in appendix A.

## Tests

```bash
python -m pytest -q        # 22 tests: audit-standard data guards, Wilson CI,
                           # falloff/trend, paired McNemar
```

The data guards are the point: they re-verify from `data/sweep/responses.json`
that the answer-colour shortcut the report says it removed is actually gone
(≥6 answer colours per family, no single colour >30%, balanced corners). CI
(`.github/workflows/ci.yml`) runs the suite on every push to `main`.

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
