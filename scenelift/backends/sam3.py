# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Segmentation backend built on SAM 3 (Meta, "SAM License").

Install:  pip install transformers
Weights:  "facebook/sam3" on Hugging Face. The repository is gated: request
          access on the model page, then log in with `hf auth login` (or set
          HF_TOKEN) before the first run.
Licence:  the SAM License is not an OSI open-source licence. Read it before any
          commercial use. scenelift does not bundle SAM 3; this file only calls it.

This wrapper uses the SAM 3 implementation that ships with Hugging Face
`transformers`, because it runs on CUDA, on Apple silicon (MPS) and on the CPU.
Meta's own `sam3` package loads the same weights but imports `triton`, which
exists for Linux with CUDA only.

SAM 3 segments by concept, so it needs words. Each view is encoded once and
then queried with every prompt in `prompts`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..types import Mask2D, View

MODEL_ID = "facebook/sam3"


class Sam3Backend:
    def __init__(self, prompts: list[str], confidence: float = 0.5, device: str | None = None,
                 model_id: str = MODEL_ID):
        import torch
        from transformers import Sam3Model, Sam3Processor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.prompts = list(prompts)
        self.confidence = float(confidence)
        try:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        except OSError as e:  # gated or missing repository, no network
            raise RuntimeError(
                f"Could not load the SAM 3 weights '{model_id}'. They are gated on Hugging Face: "
                f"request access at https://huggingface.co/{model_id}, wait until it is granted, "
                f"then log in with `hf auth login` or set HF_TOKEN. ({type(e).__name__}: "
                f"{str(e).strip().splitlines()[0]})") from e
        self._text: dict = {}  # prompt -> tokens on the device; prompts repeat for every view

    def _tokens(self, prompt: str):
        if prompt not in self._text:
            self._text[prompt] = self.processor(text=prompt, return_tensors="pt").to(self.device)
        return self._text[prompt]

    def segment(self, views: list[View]) -> list[list[Mask2D]]:
        import torch

        out = []
        for view in views:
            H, W = view.size
            masks: list[Mask2D] = []
            with torch.inference_mode():
                pixels = self.processor(images=Image.fromarray(view.image), return_tensors="pt").to(self.device)
                vision = self.model.get_vision_features(pixel_values=pixels.pixel_values)
                for prompt in self.prompts:
                    tokens = self._tokens(prompt)
                    res = self.model(vision_embeds=vision, input_ids=tokens.input_ids,
                                     attention_mask=tokens.attention_mask)
                    res = self.processor.post_process_instance_segmentation(
                        res, threshold=self.confidence, mask_threshold=0.5, target_sizes=[(H, W)])[0]
                    m = res["masks"].detach().cpu().numpy().astype(bool).reshape(-1, H, W)
                    s = res["scores"].detach().float().cpu().numpy().ravel()
                    for mask, score in zip(m, s):
                        if mask.any():
                            masks.append(Mask2D(view_index=view.index, mask=mask, label=prompt,
                                                score=float(score)))
            out.append(masks)
        return out
