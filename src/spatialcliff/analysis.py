"""Complexity-decay analysis: where does spatial reasoning break as scenes
get harder?

Core objects:

- ``ComplexityCurve``: one curve per family, accuracy over the difficulty
  axis (d0 = simplest .. d5 = hardest) with Wilson 95% CIs.
- ``Falloff``: the difficulty step at which accuracy drops by more than a
  fixed margin (the "breakpoint"), with the drop magnitude.
- ``trend``: Pearson correlation of accuracy vs difficulty, a monotonicity
  proxy that survives noisy per-point estimates.

A family is *complexity sensitive* if its curve falls with the difficulty
axis, and *complexity robust* if the curve is flat. The falloff detector
flags sharp single-step breaks; the trend captures gradual decay.

This is deliberately a small, dependency-light implementation (numpy only) so
the whole analysis is auditable. No GPU, no model code: it consumes the JSON
table produced by the sweep script.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class ComplexityCurve:
    family: str
    complexity: list[float]          # normalized difficulty (0.0 .. 1.0)
    labels: list[str]                # e.g. "d0".."d5"
    acc: list[float]
    ci_lo: list[float]
    ci_hi: list[float]
    n: list[int]

    @classmethod
    def from_rows(cls, family: str, rows: list[dict]) -> "ComplexityCurve":
        comp, labels, acc, lo, hi, n = [], [], [], [], [], []
        for r in sorted(rows, key=lambda x: x["complexity"]):
            p, l, u = _wilson(r["correct"], r["total"])
            comp.append(r["complexity"])
            labels.append(r["label"])
            acc.append(p)
            lo.append(l)
            hi.append(u)
            n.append(r["total"])
        return cls(family, comp, labels, acc, lo, hi, n)

    def falloff(self, margin: float = 0.15, min_drop: float = 0.20) -> dict | None:
        """First step (from simplest to hardest) where acc falls by >= margin
        and the running deficit vs the simplest exceeds min_drop."""
        if len(self.acc) < 2:
            return None
        base = self.acc[0]
        prev = base
        for i, a in enumerate(self.acc[1:], start=1):
            if prev - a >= margin and base - a >= min_drop:
                return {
                    "index": i,
                    "at": self.labels[i],
                    "acc_before": prev,
                    "acc_after": a,
                    "drop": prev - a,
                }
            prev = a
        return None

    def trend(self) -> float:
        """Pearson correlation of accuracy vs normalized complexity.

        A strongly negative value means accuracy decays monotonically-ish with
        complexity; ~0 means complexity-insensitive; positive means the model
        *improves* on harder scenes (which can happen after shortcut removal).
        """
        if len(self.acc) < 3:
            return 0.0
        r = np.corrcoef(self.complexity, self.acc)[0, 1]
        return float(r) if not np.isnan(r) else 0.0


@dataclass
class SweepResult:
    """One curve per family over the difficulty axis."""

    curves: dict[str, ComplexityCurve] = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "SweepResult":
        data = json.loads(path.read_text(encoding="utf-8-sig")) if hasattr(path, "read_text") else json.loads(path)
        curves: dict[str, ComplexityCurve] = {}
        for fam, rows in data["by_family"].items():
            curves[fam] = ComplexityCurve.from_rows(fam, rows)
        return cls(curves)

    def summary(self) -> dict:
        out = {}
        for fam, c in self.curves.items():
            fo = c.falloff()
            out[fam] = {
                "best_acc": max(c.acc),
                "worst_acc": min(c.acc),
                "range": max(c.acc) - min(c.acc),
                "falloff": fo,
                "trend": c.trend(),
                "n_difficulties": len(c.acc),
            }
        return out
