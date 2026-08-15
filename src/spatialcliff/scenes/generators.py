"""Deterministic procedural generation of spatial-reasoning scene images.

Each scene is a 2D top-down view rendered with PIL primitives. Four scene
families probe distinct spatial-reasoning mechanisms:

- ``relpos``   relative position: "what color is the shape to the left of the
  blue square?"  Requires locating the target and binding a property of a
  spatially-related object.
- ``occlusion`` occlusion: an object is partly hidden behind a gray square; the
  question requires reasoning *through* the occluder to recover the hidden
  object's color.
- ``lookalike`` identical-look distractors: many identical red circles, one has
  a blue dot; the model must bind the attribute to the *right instance* and
  then report its location.
- ``nearest``  nearest-neighbor: "which object is closest to the blue diamond?"
  Requires relative-distance computation over the whole scene.

The difficulty axis raises the *binding load*: more objects, more occluders,
more lookalikes, more background clutter. Ground truth is always computed from
the scene layout itself, so a model failure is a reasoning failure, never an
annotation artifact.

Design rule: the question offers a closed set of answer options and never
reveals the answer; the ground truth is one of the options, so scoring is
unambiguous. Rendering uses only PIL primitives so content is exactly
controlled.
"""
from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

W, H = 640, 640

# ---- color palette -------------------------------------------------------
COLOR_NAMES = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta"]
COLOR_RGB = {
    "red": (210, 70, 60),
    "blue": (60, 90, 210),
    "green": (60, 170, 90),
    "orange": (230, 140, 30),
    "purple": (150, 70, 200),
    "cyan": (50, 170, 190),
    "yellow": (220, 190, 40),
    "magenta": (200, 60, 160),
}
SHAPES = ["circle", "square", "triangle", "diamond"]


# ---- layout helpers --------------------------------------------------------

def _place_nonoverlap(rng: random.Random, n: int, r_min: int, r_max: int) -> list[tuple[float, float, float]]:
    """Place n circles (cx, cy, r) without overlap, inside the canvas."""
    placed: list[tuple[float, float, float]] = []
    tries = 0
    while len(placed) < n and tries < 800:
        tries += 1
        r = rng.randint(r_min, r_max)
        pad = 14
        cx = rng.uniform(r + pad, W - r - pad)
        cy = rng.uniform(r + pad, H - r - pad)
        ok = True
        for (px, py, pr) in placed:
            if math.hypot(cx - px, cy - py) < r + pr + 12:
                ok = False
                break
        if ok:
            placed.append((cx, cy, r))
    return placed


def _draw_shape(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
                shape: str, color: tuple[int, int, int]) -> None:
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color,
                     outline=(0, 0, 0), width=2)
    elif shape == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color,
                     outline=(0, 0, 0), width=2)


def _add_clutter(draw: ImageDraw.ImageDraw, rng: random.Random, n: int) -> None:
    """Irrelevant faint background glyphs the model must ignore."""
    for _ in range(n):
        cx = rng.uniform(8, W - 8)
        cy = rng.uniform(8, H - 8)
        r = rng.uniform(2, 6)
        col = tuple(rng.randint(170, 235) for _ in range(3))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)


def _pick_color(rng: random.Random, exclude: set[str]) -> str:
    pool = [c for c in COLOR_NAMES if c not in exclude]
    if not pool:
        pool = list(COLOR_NAMES)
    return rng.choice(pool)


# ---- family 1: relative position -------------------------------------------

def gen_relpos(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """The blue square is the target; exactly one other object is entirely to
    its LEFT. The model must find the object to the left and report its color.

    The answer color is randomized per scene and at least one *same-colored
    distractor* is placed outside the left region, so a model cannot solve the
    task by reporting the rarest color — it has to bind the spatial relation
    first.
    """
    rng = random.Random(seed)
    n_obj = {0: 3, 1: 4, 2: 6, 3: 8, 4: 11, 5: 15}[difficulty]
    r = {0: 34, 1: 30, 2: 26, 3: 22, 4: 18, 5: 15}[difficulty]
    n_same = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[difficulty]  # extra same-color distractors

    for _attempt in range(80):
        placed = _place_nonoverlap(rng, n_obj, r, r + 14)
        if len(placed) < n_obj:
            continue
        target_idx = rng.randrange(len(placed))
        (tx, ty, tr) = placed[target_idx]

        # objects entirely to the LEFT of the target (their right edge < target's left edge)
        left_of = [i for i, (x, y, pr) in enumerate(placed)
                   if i != target_idx and x + pr < tx - tr - 8]
        if len(left_of) != 1:
            continue
        answer_idx = left_of[0]

        # randomized answer color (never the target blue); other colors may
        # repeat, so color identity alone never identifies the answer
        answer_color = rng.choice([c for c in COLOR_NAMES if c != "blue"])

        # choose same-color distractors among non-target, non-answer objects
        candidates = [i for i in range(len(placed)) if i not in (target_idx, answer_idx)]
        n_same_actual = min(n_same, len(candidates))
        same_idx = set(rng.sample(candidates, n_same_actual)) if n_same_actual else set()

        img = Image.new("RGB", (W, H), (245, 245, 240))
        draw = ImageDraw.Draw(img)
        used_colors: set[str] = set()
        for i, (cx, cy, pr) in enumerate(placed):
            if i == target_idx:
                shape, color = "square", "blue"
            elif i == answer_idx:
                shape = rng.choice(SHAPES)
                color = answer_color
            elif i in same_idx:
                shape = rng.choice(SHAPES)
                color = answer_color
            else:
                shape = rng.choice(SHAPES)
                color = _pick_color(rng, used_colors)
                used_colors.add(color)
            _draw_shape(draw, cx, cy, pr, shape, COLOR_RGB[color])
        _add_clutter(draw, rng, {0: 0, 1: 6, 2: 12, 3: 20, 4: 30, 5: 42}[difficulty])
        return img, answer_color
    # extremely unlikely fallback: last scene wins
    return gen_relpos(seed + 10000, difficulty)


def _q_relpos() -> str:
    return ("A blue square is the target object. Exactly one other object is "
            "entirely to the left of the blue square. What is the color of that "
            "object? Answer with a single color word.")


# ---- family 2: occlusion ----------------------------------------------------

def gen_occlusion(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """One colored object is partially hidden behind a gray square; extra gray
    squares act as occluder distractors. The model must find the partially
    hidden colored object and report its color.

    The hidden object's color is randomized per scene and the same color
    appears on at least one fully visible object, so reporting a rare color is
    not a solution — the model must identify *which* instance is occluded.
    """
    rng = random.Random(seed)
    occl_r = {0: 40, 1: 36, 2: 32, 3: 28, 4: 24, 5: 20}[difficulty]
    n_gray = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty]
    n_other = {0: 1, 1: 2, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty]
    n_same = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[difficulty]  # visible same-color distractors

    for _attempt in range(80):
        # main occluder near center
        cx, cy = W // 2 + rng.randint(-50, 50), H // 2 + rng.randint(-50, 50)
        occluder_r = occl_r + 14
        # hidden colored circle offset so part of it sticks out
        off = rng.randint(occluder_r - 8, occluder_r + 6)
        hidden = (cx - off, cy + rng.randint(-10, 10), occl_r)

        # place other gray squares (non-overlapping with occluder)
        others = _place_nonoverlap(rng, n_gray + n_other - 1, occl_r - 6, occl_r + 4)
        if len(others) < n_gray + n_other - 1:
            continue

        hidden_color = rng.choice([c for c in COLOR_NAMES if c != "blue"])

        img = Image.new("RGB", (W, H), (245, 245, 240))
        draw = ImageDraw.Draw(img)
        # hidden colored circle first
        _draw_shape(draw, hidden[0], hidden[1], occl_r, "circle", COLOR_RGB[hidden_color])
        # main occluder on top
        _draw_shape(draw, cx, cy, occluder_r, "square", (150, 150, 150))
        gray_used = 1
        used_colors: set[str] = set()
        visible_colored = [i for i in range(len(others)) if i >= n_gray - 1]
        n_same_actual = min(n_same, len(visible_colored))
        same_idx = set(rng.sample(visible_colored, n_same_actual)) if n_same_actual else set()
        for i, (ox, oy, pr) in enumerate(others):
            if i < n_gray - 1:
                _draw_shape(draw, ox, oy, pr, "square", (150, 150, 150))
                gray_used += 1
            elif i in same_idx:
                shape = rng.choice(SHAPES)
                _draw_shape(draw, ox, oy, pr, shape, COLOR_RGB[hidden_color])
            else:
                shape = rng.choice(SHAPES)
                color = _pick_color(rng, used_colors)
                used_colors.add(color)
                _draw_shape(draw, ox, oy, pr, shape, COLOR_RGB[color])
        _add_clutter(draw, rng, {0: 0, 1: 8, 2: 16, 3: 24, 4: 34, 5: 46}[difficulty])
        return img, hidden_color
    return gen_occlusion(seed + 10000, difficulty)


def _q_occlusion() -> str:
    return ("Several gray squares are shown in this scene. Exactly one colored "
            "object is partially hidden behind one of the gray squares. What is "
            "the color of the partially hidden object? Answer with a single "
            "color word.")


# ---- family 3: lookalike ----------------------------------------------------

def gen_lookalike(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """Several identical red circles; exactly one carries a small blue dot. The
    model must find the right instance, then report which image corner it is
    closest to."""
    rng = random.Random(seed)
    n_same = {0: 2, 1: 3, 2: 4, 3: 5, 4: 7, 5: 9}[difficulty]
    r = {0: 30, 1: 26, 2: 22, 3: 19, 4: 16, 5: 13}[difficulty]
    n_other = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty]

    for _attempt in range(40):
        placed = _place_nonoverlap(rng, n_same + n_other, r, r + 8)
        if len(placed) < n_same + n_other:
            continue
        target_idx = rng.randrange(n_same)

        img = Image.new("RGB", (W, H), (245, 245, 240))
        draw = ImageDraw.Draw(img)
        used_colors: set[str] = set()
        for i, (cx, cy, pr) in enumerate(placed):
            if i < n_same:
                _draw_shape(draw, cx, cy, pr, "circle", COLOR_RGB["red"])
                if i == target_idx:
                    dot = max(5, pr // 2)
                    draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot],
                                 fill=COLOR_RGB["blue"])
            else:
                shape = rng.choice(SHAPES)
                color = _pick_color(rng, used_colors)
                used_colors.add(color)
                _draw_shape(draw, cx, cy, pr, shape, COLOR_RGB[color])
        _add_clutter(draw, rng, {0: 0, 1: 6, 2: 14, 3: 22, 4: 32, 5: 44}[difficulty])

        # answer: which image corner is the target circle closest to
        (tx, ty, _) = placed[target_idx]
        dists = {
            "top-left": math.hypot(tx, ty),
            "top-right": math.hypot(tx - W, ty),
            "bottom-left": math.hypot(tx, ty - H),
            "bottom-right": math.hypot(tx - W, ty - H),
        }
        zone = min(dists, key=dists.get)
        return img, zone
    return gen_lookalike(seed + 10000, difficulty)


def _q_lookalike() -> str:
    return ("Several identical red circles are shown. Exactly one of them has a "
            "small blue dot in its center. Find that circle. Which corner of "
            "the image is it closest to? Answer with one of: top-left, "
            "top-right, bottom-left, bottom-right.")


# ---- family 4: nearest neighbor ----------------------------------------------

def gen_nearest(seed: int, difficulty: int) -> tuple[Image.Image, str]:
    """A blue diamond is the anchor. The nearest object is the answer; the model
    must find the nearest object and report its color.

    The answer color is randomized per scene and at least one same-colored
    distractor sits farther away, so reporting a rare color is not a solution —
    the model must actually compute the nearest neighbor.
    """
    rng = random.Random(seed)
    n_obj = {0: 3, 1: 4, 2: 6, 3: 8, 4: 11, 5: 14}[difficulty]
    r = {0: 30, 1: 27, 2: 24, 3: 21, 4: 18, 5: 15}[difficulty]
    n_same = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[difficulty]  # farther same-color distractors

    for _attempt in range(80):
        placed = _place_nonoverlap(rng, n_obj, r, r + 8)
        if len(placed) < n_obj:
            continue
        anchor_idx = rng.randrange(len(placed))
        (ax, ay, ar) = placed[anchor_idx]
        nearest_idx = min(
            (i for i in range(len(placed)) if i != anchor_idx),
            key=lambda i: math.hypot(placed[i][0] - ax, placed[i][1] - ay),
        )
        if nearest_idx == anchor_idx:
            continue

        answer_color = rng.choice([c for c in COLOR_NAMES if c != "blue"])

        # same-color distractors among non-anchor, non-nearest objects
        candidates = [i for i in range(len(placed)) if i not in (anchor_idx, nearest_idx)]
        n_same_actual = min(n_same, len(candidates))
        same_idx = set(rng.sample(candidates, n_same_actual)) if n_same_actual else set()

        img = Image.new("RGB", (W, H), (245, 245, 240))
        draw = ImageDraw.Draw(img)
        used_colors: set[str] = set()
        for i, (cx, cy, pr) in enumerate(placed):
            if i == anchor_idx:
                shape, color = "diamond", "blue"
            elif i == nearest_idx:
                shape = rng.choice(SHAPES)
                color = answer_color
            elif i in same_idx:
                shape = rng.choice(SHAPES)
                color = answer_color
            else:
                shape = rng.choice(SHAPES)
                color = _pick_color(rng, used_colors)
                used_colors.add(color)
            _draw_shape(draw, cx, cy, pr, shape, COLOR_RGB[color])
        _add_clutter(draw, rng, {0: 0, 1: 6, 2: 14, 3: 22, 4: 32, 5: 44}[difficulty])
        return img, answer_color
    return gen_nearest(seed + 10000, difficulty)


def _q_nearest() -> str:
    return ("A blue diamond is the anchor object. Among all the other objects, "
            "which object is closest to the blue diamond? What is the color of "
            "the closest object? Answer with a single color word.")


# ---- registry ---------------------------------------------------------------

FAMILY_GENERATORS = {
    "relpos": gen_relpos,
    "occlusion": gen_occlusion,
    "lookalike": gen_lookalike,
    "nearest": gen_nearest,
}

QUESTIONS = {
    "relpos": _q_relpos(),
    "occlusion": _q_occlusion(),
    "lookalike": _q_lookalike(),
    "nearest": _q_nearest(),
}
