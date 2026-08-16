"""InternVL2.5 inference engine.

Drop-in replacement for :class:`QwenVLEngine` with the same ``ask(image,
question) -> str`` interface, so the whole sweep / analysis pipeline runs
unchanged against a second, architecturally distinct model (InternViT +
InternLM2 vs Qwen2.5-VL). The point of the second model is audit transfer:
every claim in the report is re-measured on an independent model family to
separate "how VLMs reason about space" from "how Qwen2.5-VL does".

InternVL2.5 uses dynamic-resolution image splitting, so the preprocessing
differs from Qwen's processor: ``dynamic_preprocess`` tiles the image into
448x448 blocks and appends a thumbnail, and the ``chat`` method receives the
stacked ``pixel_values`` rather than a raw PIL image. The normalization layer
in :mod:`spatialcliff.check` is model-agnostic, so answers score identically.
"""
from __future__ import annotations

import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area * ratio[0] * ratio[1] < area * best_ratio[0] * best_ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split an arbitrary-ratio image into a grid of 448px blocks + thumbnail.

    Mirrors InternVL2.5's official loader (the model repo ships no tools
    module, so the preprocessing is reimplemented here and unit-tested).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVLEngine:
    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2_5-2B",
        *,
        max_pixels: int = 448 * 448,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model_path = model_path
        self.max_pixels = max_pixels
        self._transform = _build_transform(448)

    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, *, max_new_tokens: int = 48) -> str:
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        patches = dynamic_preprocess(image, image_size=448, use_thumbnail=True)
        pixel_values = torch.stack([self._transform(p) for p in patches]).to(self.model.device)
        pixel_values = pixel_values.to(torch.bfloat16)
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        return self.model.chat(
            self.tokenizer, pixel_values, question, generation_config
        )
