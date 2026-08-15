"""Answer normalization and correctness checks.

The scene questions ask for constrained outputs (a color word, a corner label).
A small VLM will not always reply with the exact string, so we normalize
leniently per family before comparing to ground truth. Every rule is a pure
function of the model text, kept explicit so failures are inspectable rather
than hidden by a fuzzy matcher.
"""
from __future__ import annotations

import re

_COLOR_WORDS = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta", "black", "gray", "grey", "white", "brown", "pink", "teal", "lime", "indigo", "violet", "gold", "silver"]

_CORNERS = {
    "top-left": ["top left", "top-left", "upper left", "upper-left", "top left corner"],
    "top-right": ["top right", "top-right", "upper right", "upper-right", "top right corner"],
    "bottom-left": ["bottom left", "bottom-left", "lower left", "lower-left", "bottom left corner"],
    "bottom-right": ["bottom right", "bottom-right", "lower right", "lower-right", "bottom right corner"],
}


def _extract_color(text: str) -> str | None:
    t = text.lower()
    # first named color wins
    for c in _COLOR_WORDS:
        if re.search(rf"\b{c}\b", t):
            return c
    return None


def _extract_corner(text: str) -> str | None:
    t = text.lower()
    for corner, needles in _CORNERS.items():
        for n in needles:
            if n in t:
                return corner
    # fall back to tokenizing two directional words
    has_left = "left" in t
    has_right = "right" in t
    has_top = "top" in t or "upper" in t or "above" in t
    has_bottom = "bottom" in t or "lower" in t or "below" in t
    if has_left and has_top:
        return "top-left"
    if has_right and has_top:
        return "top-right"
    if has_left and has_bottom:
        return "bottom-left"
    if has_right and has_bottom:
        return "bottom-right"
    return None


def norm_color_ans(text: str) -> str | None:
    """Used by relpos, occlusion, nearest: answer is a color word."""
    return _extract_color(text)


def norm_corner_ans(text: str) -> str | None:
    """Used by lookalike: answer is a corner label."""
    return _extract_corner(text)


_NORMALIZERS = {
    "relpos": norm_color_ans,
    "occlusion": norm_color_ans,
    "nearest": norm_color_ans,
    "lookalike": norm_corner_ans,
}


def normalize(family: str, raw: str) -> str | None:
    return _NORMALIZERS[family](raw)


def is_correct(family: str, raw: str, answer: str) -> bool:
    n = normalize(family, raw)
    if n is None:
        return False
    return n == answer
