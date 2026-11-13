# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Geometry backend built on MapAnything (Meta, Apache 2.0 code).

Install:  git clone https://github.com/facebookresearch/map-anything && pip install -e map-anything
Weights:  "facebook/map-anything-apache" is the Apache 2.0 checkpoint and is the
          default here. "facebook/map-anything" is the other public checkpoint;
          it is not the Apache one, so read its model card before using it.

The call sequence follows the "Image-Only Inference" section of the MapAnything
README (checked November 2026). This wrapper has not been run by the authors of
this file on real weights; run tests/test_backends_live.py after installing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import Frame, View


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class MapAnythingBackend:
    def __init__(self, model_id: str = "facebook/map-anything-apache", device: str | None = None,
                 minibatch_size: int | None = None):
        import torch
        from mapanything.models import MapAnything

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.minibatch_size = minibatch_size
        self.model = MapAnything.from_pretrained(model_id).to(device).eval()

    def infer(self, frames: list[Frame]) -> list[View]:
        import torch
        from mapanything.utils.image import load_images

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in frames:
                p = Path(tmp) / f"{f.index:05d}.png"
                Image.fromarray(f.image).save(p)
                paths.append(str(p))
            views_in = load_images(paths)
            with torch.no_grad():
                preds = self.model.infer(
                    views_in,
                    memory_efficient_inference=True,
                    minibatch_size=self.minibatch_size,
                    use_amp=self.device == "cuda",
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    apply_confidence_mask=False,
                )
        out = []
        for f, pred in zip(frames, preds):
            depth = _np(pred["depth_z"])[0, ..., 0].astype(np.float32)
            conf = _np(pred["conf"])[0].astype(np.float32)
            valid = _np(pred["mask"])[0, ..., 0] > 0.5
            img = _np(pred["img_no_norm"])[0]
            if img.max() <= 1.5:
                img = img * 255.0
            # the model's confidence is unbounded; squash it to [0, 1] per view
            c_lo, c_hi = np.percentile(conf, 2), np.percentile(conf, 98)
            conf = np.clip((conf - c_lo) / max(c_hi - c_lo, 1e-6), 0.0, 1.0)
            out.append(View(index=f.index, image=np.clip(img, 0, 255).astype(np.uint8), depth=depth,
                            K=_np(pred["intrinsics"])[0].astype(np.float64),
                            cam2world=_np(pred["camera_poses"])[0].astype(np.float64),
                            conf=conf, valid=valid & np.isfinite(depth) & (depth > 0),
                            source=f.source))
        return out
