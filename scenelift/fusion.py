# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 7: fuse each object's multi-view observations into one clean point cloud,
and collect everything that is not an object into the background."""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .geometry import largest_cluster_mask, statistical_outlier_mask, unproject, voxel_downsample
from .scale import usable_mask
from .types import Instance3D, Mask2D, View


def fuse_instance(inst: Instance3D, cfg: Config):
    """Returns (points, colors) of the measured surface of one object, or None."""
    pts = np.concatenate([c[0] for c in inst.chunks])
    cols = np.concatenate([c[1] for c in inst.chunks])
    conf = np.concatenate([c[2] for c in inst.chunks])
    pts, cols, _ = voxel_downsample(pts, cfg.fuse_voxel, cols, weights=conf)
    if len(pts) < cfg.min_object_points:
        return None
    keep = statistical_outlier_mask(pts, cfg.sor_k, cfg.sor_std)
    pts, cols = pts[keep], cols[keep]
    big = largest_cluster_mask(pts, max(0.05, 5 * cfg.fuse_voxel))
    if big.mean() >= 0.6:  # drop stray fragments, but never split a real object
        pts, cols = pts[big], cols[big]
    if len(pts) < cfg.min_object_points:
        return None
    return pts, cols


def fuse_background(views: list[View], masks_per_view: list[list[Mask2D]], cfg: Config):
    """Everything outside the (dilated) object masks, voxel-averaged."""
    acc_p, acc_c = [], []
    for view, masks in zip(views, masks_per_view):
        occupied = np.zeros(view.size, dtype=np.uint8)
        for m in masks:
            occupied |= m.mask.astype(np.uint8)
        occupied = cv2.dilate(occupied, np.ones((7, 7), np.uint8)).astype(bool)
        keep = usable_mask(view, cfg) & ~occupied
        pts, v, u = unproject(view.depth, view.K, view.cam2world, keep)
        p, c, _ = voxel_downsample(pts, cfg.background_voxel, view.image[v, u])
        acc_p.append(p)
        acc_c.append(c)
    if not acc_p:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    p, c, _ = voxel_downsample(np.concatenate(acc_p), cfg.background_voxel, np.concatenate(acc_c))
    return p, c
