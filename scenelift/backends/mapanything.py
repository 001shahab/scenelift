# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Geometry backend built on MapAnything (Meta, Apache 2.0 code).

Install:  pip install git+https://github.com/facebookresearch/map-anything.git
Weights:  "facebook/map-anything-apache" is the Apache 2.0 checkpoint and is the
          default here. "facebook/map-anything" is the other public checkpoint;
          it is not the Apache one, so read its model card before using it.

The call sequence follows the "Image-Only Inference" section of the MapAnything
README (checked November 2026). Run on real weights on Apple silicon (MPS); see
docs/BACKENDS.md for what was observed.

Two things here come from that first contact with real footage:

  * One camera, one set of intrinsics. The frames of a video come from one
    lens, but the model estimates a focal length per frame, and on frames that
    show little but a blank wall the estimate was off by almost a factor of
    two. For a video the model is therefore run twice: once to estimate the
    intrinsics, and once with the confidence-weighted median of those
    estimates given to every frame.
  * Confidence is a rank, not a value. The raw confidence is unbounded and its
    level differs a lot between the frames of one clip. What is comparable is
    the order within a frame, so the weakest `conf_floor` of each frame (10
    percent, MapAnything's own default) is mapped below `conf_at_floor`, the
    value at which the pipeline drops a pixel (Config.min_conf).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import Frame, View


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def rank_confidence(conf: np.ndarray, floor: float = 0.10, at_floor: float = 0.3) -> np.ndarray:
    """Map raw confidence to 0..1 by its rank within the image: the weakest
    `floor` of the pixels land below `at_floor`, the rest between it and 1."""
    order = np.argsort(conf, axis=None, kind="stable")
    rank = np.empty(conf.size, dtype=np.float64)
    rank[order] = (np.arange(conf.size) + 0.5) / conf.size
    out = np.where(rank < floor, rank / floor * at_floor,
                   at_floor + (rank - floor) / (1.0 - floor) * (1.0 - at_floor))
    return out.reshape(conf.shape).astype(np.float32)


def shared_intrinsics(Ks: list, weights: list) -> np.ndarray:
    """One intrinsics matrix for all frames of one camera: the weighted median
    of the per-frame estimates, entry by entry."""
    Ks, w = np.asarray(Ks, dtype=np.float64), np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)

    def wmedian(x):
        order = np.argsort(x)
        cum = np.cumsum(w[order]) / w.sum()
        return float(x[order][min(int(np.searchsorted(cum, 0.5)), len(x) - 1)])

    K = np.eye(3)
    K[0, 0], K[1, 1] = wmedian(Ks[:, 0, 0]), wmedian(Ks[:, 1, 1])
    K[0, 2], K[1, 2] = wmedian(Ks[:, 0, 2]), wmedian(Ks[:, 1, 2])
    return K


def one_camera(frames: list[Frame]) -> bool:
    """True for the frames of one video: same lens, same zoom, same size."""
    if len(frames) < 2 or any(f.timestamp is None or not f.source for f in frames):
        return False
    videos = {f.source.rsplit("#", 1)[0] for f in frames}
    return len(videos) == 1 and len({f.image.shape for f in frames}) == 1


class MapAnythingBackend:
    def __init__(self, model_id: str = "facebook/map-anything-apache", device: str | None = None,
                 minibatch_size: int | None = None, share_intrinsics: bool | None = None,
                 conf_floor: float = 0.10, conf_at_floor: float = 0.3):
        """share_intrinsics: give all frames the same intrinsics. None decides
        per input: yes for the frames of one video, no for separate photos."""
        import torch
        from mapanything.models import MapAnything

        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.minibatch_size = minibatch_size
        self.share_intrinsics = share_intrinsics
        self.conf_floor = conf_floor
        self.conf_at_floor = conf_at_floor
        self.model = MapAnything.from_pretrained(model_id).to(device).eval()

    def _run(self, paths: list[str], K: np.ndarray | None = None):
        import torch
        from mapanything.utils.image import load_images

        views_in = load_images(paths)
        if K is not None:
            for v in views_in:
                v["intrinsics"] = torch.tensor(K, dtype=torch.float32)[None]
        with torch.no_grad():
            return self.model.infer(
                views_in,
                memory_efficient_inference=True,
                minibatch_size=self.minibatch_size,
                use_amp=self.device == "cuda",
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )

    def infer(self, frames: list[Frame]) -> list[View]:
        share = one_camera(frames) if self.share_intrinsics is None else self.share_intrinsics
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in frames:
                p = Path(tmp) / f"{f.index:05d}.png"
                Image.fromarray(f.image).save(p)
                paths.append(str(p))
            preds = self._run(paths)
            if share and len(frames) > 1:
                K = shared_intrinsics([_np(p["intrinsics"])[0] for p in preds],
                                      [float(np.median(_np(p["conf"]))) for p in preds])
                preds = self._run(paths, K)
        out = []
        for f, pred in zip(frames, preds):
            depth = _np(pred["depth_z"])[0, ..., 0].astype(np.float32)
            valid = _np(pred["mask"])[0, ..., 0] > 0.5
            img = _np(pred["img_no_norm"])[0]
            if img.max() <= 1.5:
                img = img * 255.0
            conf = rank_confidence(_np(pred["conf"])[0], self.conf_floor, self.conf_at_floor)
            out.append(View(index=f.index, image=np.clip(img, 0, 255).astype(np.uint8), depth=depth,
                            K=_np(pred["intrinsics"])[0].astype(np.float64),
                            cam2world=_np(pred["camera_poses"])[0].astype(np.float64),
                            conf=conf, valid=valid & np.isfinite(depth) & (depth > 0),
                            source=f.source))
        return out
