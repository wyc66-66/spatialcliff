"""Qwen2.5-VL inference engine.

Wraps the standard HF Qwen2.5-VL processor + model flow with a minimal
zero-shot QA interface. Image input is capped by ``max_pixels`` so each
scene stays within a small number of vision tokens — the scenes are simple
geometric layouts, and the stress axis is scene complexity, not image
resolution. The cap also keeps inference fast enough for a 480-scene sweep
on a laptop GPU.
"""
from __future__ import annotations

import torch
from PIL import Image


class QwenVLEngine:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        *,
        max_pixels: int = 448 * 448,
    ):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model_path = model_path
        self.max_pixels = max_pixels

    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, *, max_new_tokens: int = 48) -> str:
        # cap the input resolution so a complex scene doesn't blow up the
        # vision-token count (and the sweep time) on a laptop GPU
        if image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
