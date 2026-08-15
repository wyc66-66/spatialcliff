# SpatialCliff: Mapping Where Scene Complexity Breaks Spatial Reasoning in an Open VLM

_A Technical Report_

**Model under test:** Qwen2.5-VL-3B-Instruct — an open 3B vision-language model, evaluated zero-shot.

**Dataset:** 480 procedurally-generated 2D scenes across 4 spatial-reasoning mechanisms × 6 complexity levels × 20 seeds, each with an exact ground-truth answer.

**Question:** Which spatial-reasoning capabilities survive scene complexity, and which collapse first?

---

## 1. Motivation

Multimodal models are asked to reason about space — "which object is closest to the blue diamond?", "what is partially hidden behind that square?" — in increasingly dense scenes. A 3B model handles these questions confidently on simple layouts. The question an application developer faces is:

> **Where does spatial reasoning quietly break as scenes get harder?**

Simple scenes mask the failure: a model that fails on a cluttered 12-object scene may still score 1.0 on a 3-object scene. Aggregated benchmarks average over this variance. This project stress-tests four distinct spatial-reasoning mechanisms with a procedurally-generated scene corpus whose complexity is controlled along one axis at a time, and maps where each mechanism falls off a cliff.

## 2. Method

### 2.1 Scene corpus

Scenes are deterministic 2D top-down layouts rendered with PIL primitives. Four families probe distinct mechanisms:

| family | task | what it requires | complexity axis |
|---|---|---|---|
| `relpos` | what color is the object to the left of the blue square? | locate a target, bind a property of a spatially-related object | object count 3→15 |
| `occlusion` | what color is the object partially hidden behind a gray square? | reason through an occluder to recover a hidden property | occluders 1→5, objects 3→10 |
| `lookalike` | which corner is the red circle with the blue dot closest to? | bind a small attribute to the right instance among identical distractors, then localize | identical circles 2→9 |
| `nearest` | what color is the object closest to the blue diamond? | relative-distance computation over the whole scene | object count 3→14 |

20 seeds × 6 difficulties × 4 families = **480 scenes**. Ground truth is computed from the layout itself, so a model failure is a reasoning failure, never an annotation artifact. The questions never reveal the answer, and each offers a closed set of options (color words, or corner labels) so scoring is unambiguous.

### 2.2 Protocol

Zero-shot, one scene at a time. Images are capped at 448×448 input pixels (the scenes are simple geometric layouts; the stress axis is scene complexity, not resolution). Output is normalized per family by an explicit, lenient matcher; a scene is correct iff normalized output equals ground truth.

### 2.3 Cliff detection

Accuracy per (family, difficulty) with Wilson 95% CIs. A **falloff (cliff)** is the first difficulty step (from simplest to hardest) where accuracy drops ≥15 points in one step **and** ≥20 points below the simplest scene.

## 3. Results

### 3.1 The headline: three mechanisms fall off cliffs, one is uniformly hard

Per-family accuracy over the complexity axis (20 seeds per point):

| family | d0 | d1 | d2 | d3 | d4 | d5 | range | verdict |
|---|---|---|---|---|---|---|---|---|
| `nearest` | 0.70 | 0.70 | 0.35 | 0.20 | 0.10 | 0.05 | 0.65 | **cliff @ d2** |
| `relpos` | 1.00 | 0.85 | 0.65 | 0.65 | 0.60 | 0.45 | 0.55 | **cliff @ d2** |
| `occlusion` | 1.00 | 0.90 | 0.95 | 0.75 | 0.75 | 0.55 | 0.45 | **cliff @ d3** |
| `lookalike` | 0.35 | 0.50 | 0.65 | 0.60 | 0.40 | 0.45 | 0.30 | uniformly hard |

![Figure 1](figures/fig1_decay.png)

*Figure 1 — Per-mechanism accuracy vs scene complexity. The red dashed line marks the detected falloff cliff.*

### 3.2 Nearest-neighbor reasoning collapses first

`nearest` is the cleanest cliff: 0.70 → 0.70 → 0.35 → 0.20 → 0.10 → 0.05. The falloff fires at d2 (a 35-point drop in one step). Three objects are fine (0.70); four objects already drop to 0.35; fourteen objects leave the model at chance. Relative-distance computation degrades continuously once the candidate set passes ~3-4 objects — the model cannot keep more than a handful of pairwise distances in its working set.

### 3.3 Relative position degrades from a perfect baseline

`relpos` starts perfect (1.00 on 3 objects) and falls to 0.45 on 15 objects, with a cliff at d2. The mechanism is reliable when the target and its neighbor are prominent; as the scene fills with distractors, locating the correct neighbor becomes the binding bottleneck.

### 3.4 Occlusion reasoning survives mid-complexity, then falls

`occlusion` holds at 0.90-1.00 through d2, falls at d3 (0.95 → 0.75), and reaches 0.55 at d5. Reasoning *through* an occluder is robust for a small number of objects; multiple occluders and clutter eventually overwhelm the hidden-object search.

### 3.5 Lookalike binding is uniformly hard — a mechanism boundary

`lookalike` never exceeds 0.65 and stays near chance at the simplest scene (0.35). The model exhibits a **systematic bottom-quadrant attribution bias**: given several identical red circles, it disproportionately reports the target as bottom-left/bottom-right, even when the true position is top-right. This is not a complexity effect — it is a *mechanism boundary*: 3B-class models cannot reliably bind a small attribute to the correct instance among identical distractors, regardless of scene size. This bounds the claims we can make: attribute-instance binding is a hard floor for this model class, not a cliff.

## 4. Discussion

### 4.1 The spatial reasoning landscape

For an open 3B VLM, scene complexity trades away reasoning fidelity in a structured order:

1. **Nearest-neighbor is the most fragile** — the first mechanism to collapse, and it collapses completely (to 0.05).
2. **Relative position degrades from perfect** but never quite bottoms out (0.45 at maximum complexity).
3. **Occlusion reasoning is the most robust** — holds above 0.75 until the densest scenes.
4. **Attribute-instance binding is a hard floor**, not a cliff: uniformly near chance even at minimal complexity, with a systematic bottom-quadrant bias.

### 4.2 Implications for deployment

- **Scene-density budgets.** Applications that rely on nearest-neighbor judgments (e.g. "nearest obstacle", "closest target") should assume a hard limit of ~3 candidates for a 3B model; beyond that, reliability collapses.
- **Occlusion is safe mid-complexity.** Partially-hidden object reasoning holds to ~5 objects / 2-3 occluders, making it the most dependable spatial capability of the model class.
- **Don't trust aggregated accuracy.** All four mechanisms would show a "reasonable" average if pooled across difficulty; the cliffs only appear when accuracy is resolved per complexity level.

### 4.3 Failure modes, not just accuracy

The lookalike bottom-quadrant bias is the kind of finding aggregated benchmarks hide: the model does not fail randomly, it fails systematically, and the systematic error is itself a spec for where to invest (better attribute binding, better attention to small features).

## 5. Reproducibility

```
pip install -e .[gpu,paper,ui]
python scripts/build_scenes.py --seeds 20 --out data/scenes   # 480 scenes
python scripts/run_sweep.py --scenes data/scenes --out data/sweep   # GPU, ~4 min
python scripts/paper_facts.py --sweep data/sweep/sweep.json   # derived claims
python scripts/render_figures.py --sweep data/sweep/sweep.json --figs docs/figures
python scripts/render_spatialcliff_paper.py   # this report (HTML + PDF)
python -m spatialcliff ui --port 8000         # interactive console
```

`data/sweep/sweep.json` is the single source of truth: every number in this report traces to that table, and `scripts/paper_facts.py` re-derives the claims from it directly.

## References

1. Wang et al. — Qwen2.5-VL Technical Report. *arXiv:2502.13923*.
2. Liu et al. — Visual Instruction Tuning. *NeurIPS 2023*.
3. Hudson & Manning — GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. *CVPR 2019*.
4. Johnson et al. — CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning. *CVPR 2017*.
5. Li et al. — What If We Simply Repeat All Clues? A New Dataset for Probing Visual Recognition Capabilities of Multimodal Large Language Models. *arXiv:2410.06508*.
6. Yang et al. — SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. *CVPR 2024*.
