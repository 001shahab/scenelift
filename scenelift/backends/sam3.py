# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Segmentation backend built on SAM 3 (Meta, "SAM License").

Install:  git clone https://github.com/facebookresearch/sam3 && pip install -e sam3
Licence:  the SAM License is not an OSI open-source licence. Read it before any
          commercial use. scenelift does not bundle SAM 3; this file only calls it.

SAM 3 segments by concept, so it needs words. Each view is queried with every
prompt in Config.prompts. The call sequence follows the SAM 3 README (checked
November 2026); this wrapper has not been run on real weights by its authors.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..types import Mask2D, View


class Sam3Backend:
    def __init__(self, prompts: list[str], confidence: float = 0.5, device: str | None = None):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.prompts = list(prompts)
        model = build_sam3_image_model()
        kwargs = {"confidence_threshold": confidence}
        if device is not None:
            kwargs["device"] = device
        self.processor = Sam3Processor(model, **kwargs)

    def segment(self, views: list[View]) -> list[list[Mask2D]]:
        out = []
        for view in views:
            H, W = view.size
            state = self.processor.set_image(Image.fromarray(view.image))
            masks: list[Mask2D] = []
            for prompt in self.prompts:
                res = self.processor.set_text_prompt(state=state, prompt=prompt)
                m = res["masks"]
                s = res["scores"]
                m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
                s = s.detach().float().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
                m = m.reshape(-1, m.shape[-2], m.shape[-1]).astype(bool)
                for mask, score in zip(m, s.ravel()):
                    if mask.shape != (H, W):
                        continue
                    masks.append(Mask2D(view_index=view.index, mask=mask, label=prompt, score=float(score)))
                self.processor.reset_all_prompts(state)
            out.append(masks)
        return out
