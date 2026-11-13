# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Prompting: pick an object by clicking a pixel, drawing a box, or typing text."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import project


def render_id_map(scene, camera_index: int, downscale: int = 4) -> tuple[np.ndarray, list]:
    """Small image where each pixel holds the index of the object visible there.

    Returns (id_map, ids): id_map is int32 with -1 for background, values index
    into ids. Rendering is point splatting with a z-buffer; it is only used to
    resolve clicks, not for display.
    """
    cam = scene.cameras[camera_index]
    H, W = cam["size"]
    h, w = max(1, H // downscale), max(1, W // downscale)
    K = np.array(cam["K"], dtype=float).copy()
    K[:2] /= downscale
    world2cam = np.linalg.inv(np.array(cam["cam2world"], dtype=float))
    zbuf = np.full(h * w, np.inf)
    idmap = np.full(h * w, -1, dtype=np.int32)
    ids = scene.ids
    layers = [(-1, scene.background_points)] + [
        (k, scene.objects[i].surface_samples(15000)[0]) for k, i in enumerate(ids)]
    for k, pts in layers:
        if len(pts) == 0:
            continue
        u, v, z = project(pts, K, world2cam)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        flat, zz = vi[ok] * w + ui[ok], z[ok]
        order = np.argsort(-zz)  # far first, so the nearest write wins
        flat, zz = flat[order], zz[order]
        closer = zz < zbuf[flat] + 0.03
        zbuf[flat[closer]] = np.minimum(zbuf[flat[closer]], zz[closer])
        idmap[flat[closer]] = k
    idmap = idmap.reshape(h, w)
    # close the pin holes between splatted points without growing over the background layer
    filled = cv2.dilate((idmap + 1).astype(np.uint16), np.ones((3, 3), np.uint8)).astype(np.int32) - 1
    idmap = np.where(idmap >= 0, idmap, np.where(zbuf.reshape(h, w) == np.inf, filled, idmap))
    return idmap, ids


def by_click(scene, camera_index: int, u: float, v: float, downscale: int = 4) -> Optional[str]:
    """Object under pixel (u, v) of one input frame, or None for background."""
    idmap, ids = render_id_map(scene, camera_index, downscale)
    r, c = int(v) // downscale, int(u) // downscale
    r, c = np.clip(r, 0, idmap.shape[0] - 1), np.clip(c, 0, idmap.shape[1] - 1)
    win = idmap[max(0, r - 1): r + 2, max(0, c - 1): c + 2].ravel()
    win = win[win >= 0]
    if len(win) == 0:
        return None
    return ids[int(np.bincount(win).argmax())]


def by_box(scene, camera_index: int, box, downscale: int = 4, min_fraction: float = 0.15) -> list:
    """Objects inside box = (u0, v0, u1, v1), largest share of the box first."""
    idmap, ids = render_id_map(scene, camera_index, downscale)
    u0, v0, u1, v1 = (int(x) // downscale for x in box)
    win = idmap[max(0, v0): v1 + 1, max(0, u0): u1 + 1].ravel()
    if len(win) == 0:
        return []
    hits = win[win >= 0]
    if len(hits) == 0:
        return []
    counts = np.bincount(hits, minlength=len(ids))
    order = np.argsort(-counts)
    return [ids[i] for i in order if counts[i] >= min_fraction * len(win)]


def by_text(scene, text: str) -> list:
    """Objects whose label matches the text, e.g. 'chair' or 'the red mug'.

    Matching here is by label only. Attribute words such as colours are handled
    by the segmenter at reconstruction time (pass them as prompts), or by a
    language model on top of query.TOOLS.
    """
    return scene.find(text)
