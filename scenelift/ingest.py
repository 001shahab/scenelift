# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 1: turn a video, a folder or a list of photos into a small set of sharp frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Config
from .types import Frame

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on a small grey copy. Higher means sharper."""
    g = cv2.cvtColor(_resize(img, 480), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _median_flow(a: np.ndarray, b: np.ndarray) -> float:
    """Median sparse optical flow in pixels between two frames, at 320 px width."""
    def prep(x):
        s = 320.0 / x.shape[1]
        return cv2.cvtColor(cv2.resize(x, (320, int(round(x.shape[0] * s)))), cv2.COLOR_RGB2GRAY)

    ga, gb = prep(a), prep(b)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 10:
        return np.inf  # nothing to track: treat as a new viewpoint
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return np.inf
    return float(np.median(np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)))


def load_images(paths: Sequence[Union[str, os.PathLike]], cfg: Config) -> list[Frame]:
    frames = []
    for i, p in enumerate(paths):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # honour phone rotation
        frames.append(Frame(index=i, image=_resize(np.asarray(im), cfg.max_image_side),
                            source=str(p)))
    return frames


def extract_keyframes(video: Union[str, os.PathLike], cfg: Config) -> list[Frame]:
    """Pick up to cfg.max_frames frames that are sharp and show a new viewpoint.

    Pass 1 reads every cfg.frame_stride-th frame and scores sharpness.
    Pass 2 splits time into equal windows and keeps the sharpest frame of each.
    Pass 3 drops a frame when the camera hardly moved since the last kept one.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cands = []  # (frame_no, sharpness, image)
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if n % cfg.frame_stride == 0:
            ok, bgr = cap.retrieve()
            if ok:
                rgb = _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cfg.max_image_side)
                cands.append((n, sharpness(rgb), rgb))
        n += 1
    cap.release()
    if not cands:
        raise IOError(f"no frames decoded from {video}")

    n_win = min(cfg.max_frames * 2, len(cands))
    edges = np.linspace(0, len(cands), n_win + 1).astype(int)
    best = [max(cands[a:b], key=lambda c: c[1]) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    kept = [best[0]]
    for c in best[1:]:
        if _median_flow(kept[-1][2], c[2]) >= cfg.min_flow_px:
            kept.append(c)
    if len(kept) > cfg.max_frames:
        sel = np.linspace(0, len(kept) - 1, cfg.max_frames).round().astype(int)
        kept = [kept[i] for i in sel]
    return [Frame(index=i, image=img, source=f"{video}#{no}", timestamp=no / fps)
            for i, (no, _, img) in enumerate(kept)]


def load(source, cfg: Config) -> list[Frame]:
    """Accepts a video file, a folder of images, one image, or a list of image paths."""
    if isinstance(source, (list, tuple)):
        return load_images(source, cfg)
    p = Path(source)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise IOError(f"no images found in {p}")
        if len(files) > cfg.max_frames:
            sel = np.linspace(0, len(files) - 1, cfg.max_frames).round().astype(int)
            files = [files[i] for i in sel]
        return load_images(files, cfg)
    if p.suffix.lower() in VIDEO_EXT:
        return extract_keyframes(p, cfg)
    if p.suffix.lower() in IMAGE_EXT:
        return load_images([p], cfg)
    raise IOError(f"unsupported input: {source}")
