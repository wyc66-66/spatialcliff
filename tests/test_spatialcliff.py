"""Tests for SpatialCliff: analysis functions + the audit-standard data guards.

The project's entire claim is "we removed the answer-color shortcut from the
benchmark" (paper §2.1). That claim must be *tested*, not asserted: these tests
re-verify the two audit guarantees straight from the archived raw responses:
(a) the answer colour is spread over >= 6 colours per color family, with no
single colour > 30% of answers; (b) corner answers (lookalike) are balanced.
They also unit-test the statistical machinery (Wilson, falloff, trend, and the
paired McNemar test that backs the relpos d1 claim).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatialcliff.analysis import ComplexityCurve, SweepResult, paired_mcnemar, _wilson
from spatialcliff.check import is_correct, normalize

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "data" / "sweep" / "sweep.json"
RESPONSES = ROOT / "data" / "sweep" / "responses.json"


# --------------------------------------------------------------------------- #
# Answer normalization
# --------------------------------------------------------------------------- #
class TestNorm:
    def test_color_answer(self):
        assert normalize("relpos", "the object is RED") == "red"
        assert normalize("nearest", "it is cyan") == "cyan"
        assert normalize("occlusion", "yellow, probably") == "yellow"
        assert normalize("relpos", "no idea") is None

    def test_corner_answer(self):
        assert normalize("lookalike", "top left") == "top-left"
        assert normalize("lookalike", "it is the upper-right corner") == "top-right"
        assert normalize("lookalike", "bottom left") == "bottom-left"
        assert normalize("lookalike", "lower right") == "bottom-right"

    def test_is_correct(self):
        assert is_correct("relpos", "red", "red") is True
        assert is_correct("relpos", "blue", "red") is False
        assert is_correct("lookalike", "top-left", "top-left") is True


# --------------------------------------------------------------------------- #
# Statistical machinery
# --------------------------------------------------------------------------- #
class TestWilson:
    def test_zero_n(self):
        assert _wilson(0, 0) == (0.0, 0.0, 0.0)

    def test_contains_p(self):
        p, lo, hi = _wilson(20, 40)
        assert lo <= 0.5 <= hi

    def test_narrow_at_big_n(self):
        _, lo, hi = _wilson(800, 1000)
        assert hi - lo < 0.06


class TestFalloff:
    def _curve(self, accs, labels=None):
        labels = labels or [f"d{i}" for i in range(len(accs))]
        comp = [i / max(1, len(accs) - 1) for i in range(len(accs))]
        return ComplexityCurve(
            family="t", complexity=comp, labels=labels, acc=accs,
            ci_lo=[0.0] * len(accs), ci_hi=[1.0] * len(accs), n=[40] * len(accs),
        )

    def test_sharp_falloff_detected(self):
        c = self._curve([0.90, 0.85, 0.40, 0.35])
        fo = c.falloff()
        assert fo is not None
        assert fo["at"] == "d2"
        assert fo["drop"] == pytest.approx(0.45)

    def test_gradual_decay_no_falloff(self):
        c = self._curve([0.95, 0.85, 0.75, 0.65, 0.55])
        assert c.falloff() is None

    def test_flat_no_falloff(self):
        c = self._curve([0.5, 0.5, 0.5])
        assert c.falloff() is None

    def test_trend_negative_on_decay(self):
        c = self._curve([0.9, 0.7, 0.5, 0.3])
        assert c.trend() < -0.9

    def test_trend_zero_on_flat(self):
        c = self._curve([0.5, 0.5, 0.5, 0.5])
        assert abs(c.trend()) < 0.1


class TestMcNemar:
    def test_significant_direction(self):
        # relpos-like: 40 pairs; 11 got worse (d0 ok -> d1 fail), 1 improved
        # (d0 fail -> d1 ok), 28 unchanged correct, 0 unchanged wrong
        first = [True] * 39 + [False]
        second = [True] * 28 + [False] * 11 + [True]
        m = paired_mcnemar(first, second)
        assert m["b"] == 11
        assert m["c"] == 1
        assert m["n_discordant"] == 12
        assert m["p"] == pytest.approx(0.0063, abs=0.001)

    def test_no_change_gives_p1(self):
        m = paired_mcnemar([True] * 10, [True] * 10)
        assert m["p"] == 1.0

    def test_balanced_improvement_not_significant(self):
        first = [True, True, False, False]
        second = [True, False, True, False]
        m = paired_mcnemar(first, second)
        assert m["b"] == m["c"] == 1
        assert m["p"] > 0.05

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError):
            paired_mcnemar([True], [True, False])


# --------------------------------------------------------------------------- #
# Data guards: the audit standard, verified from the archived responses
# --------------------------------------------------------------------------- #
class TestAuditStandard:
    @pytest.fixture(scope="class")
    def responses(self):
        return json.loads(RESPONSES.read_text(encoding="utf-8"))

    def test_total_scenes(self, responses):
        assert len(responses) == 960

    def test_color_answers_spread_over_six_colours(self, responses):
        """paper §2.1: answer colour is sampled independently per scene and
        spread over >= 6 colours for every colour-bearing family."""
        for fam in ("relpos", "occlusion", "nearest"):
            answers = [r["answer"] for r in responses if r["family"] == fam]
            n_colours = len(set(answers))
            assert n_colours >= 6, f"{fam}: only {n_colours} answer colours"

    def test_no_single_answer_colour_dominates(self, responses):
        """paper §2.1: no single colour accounts for > 30% of answers within a
        colour-bearing family (30% is the shortcut-detection threshold)."""
        from collections import Counter

        for fam in ("relpos", "occlusion", "nearest"):
            counts = Counter(r["answer"] for r in responses if r["family"] == fam)
            n = sum(counts.values())
            top = counts.most_common(1)[0][1]
            assert top / n <= 0.30, f"{fam}: {top}/{n} on one colour"

    def test_lookalike_corners_balanced(self, responses):
        """lookalike answers four corners; no corner may dominate the corpus."""
        from collections import Counter

        corners = Counter(r["answer"] for r in responses if r["family"] == "lookalike")
        assert set(corners) == {"top-left", "top-right", "bottom-left", "bottom-right"}
        n = sum(corners.values())
        assert max(corners.values()) / n <= 0.5

    def test_every_response_has_exact_ground_truth(self, responses):
        for r in responses:
            assert r["answer"]
            assert r["family"] in ("relpos", "occlusion", "nearest", "lookalike")


class TestSweepData:
    def test_loads_and_summary(self):
        result = SweepResult.load(SWEEP)
        s = result.summary()
        assert set(s) == {"lookalike", "nearest", "occlusion", "relpos"}
        # audited sweep: only relpos keeps a falloff
        assert s["relpos"]["falloff"] is not None
        assert s["nearest"]["falloff"] is None
        assert s["occlusion"]["falloff"] is None
        assert s["lookalike"]["falloff"] is None

    def test_relpos_falloff_location(self):
        result = SweepResult.load(SWEEP)
        fo = result.summary()["relpos"]["falloff"]
        assert fo["at"] == "d1"
        assert fo["drop"] == pytest.approx(0.25, abs=0.02)
