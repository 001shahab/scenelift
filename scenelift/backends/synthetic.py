# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""A fully synthetic scene and matching fake backends.

This exists so the whole pipeline can be tested on any laptop in seconds, with
exact ground truth, no GPU and no model weights. The room holds a table, a
small box standing on the table, and a cabinet turned by 20 degrees. Depth is
ray-cast analytically, then corrupted with a different scale error per view
plus pixel noise, which is the failure the real backbones show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from ..geometry import rot_z
from ..types import CompletedShape, Frame, Mask2D, View


@dataclass
class Box:
    name: str
    center: np.ndarray
    size: np.ndarray  # full extents along the box's own x, y, z
    yaw: float
    color: tuple


def default_boxes() -> list[Box]:
    return [
        Box("table", np.array([0.0, 0.0, 0.375]), np.array([1.2, 0.7, 0.75]), 0.0, (150, 110, 70)),
        Box("box", np.array([0.2, 0.1, 0.875]), np.array([0.25, 0.2, 0.25]), 0.0, (200, 40, 40)),
        Box("cabinet", np.array([-1.3, 1.2, 0.5]), np.array([0.5, 0.4, 1.0]), np.deg2rad(20), (60, 90, 160)),
    ]


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = pos
    return T


class SyntheticWorld:
    def __init__(self, n_views: int = 10, size: tuple = (240, 320), scale_jitter: float = 0.05,
                 depth_noise: float = 0.003, seed: int = 0, boxes: Optional[list] = None):
        self.rng = np.random.default_rng(seed)
        self.boxes = boxes or default_boxes()
        self.H, self.W = size
        f = 0.9 * self.W
        self.K = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])
        angles = np.deg2rad(np.linspace(-110, 20, n_views))
        self.poses_true = [look_at(np.array([-0.3 + 2.4 * np.cos(a), 0.3 + 2.4 * np.sin(a), 1.6]),
                                   np.array([-0.3, 0.3, 0.45])) for a in angles]
        # like a real backbone, report poses relative to the first camera
        self.world_from_true = np.linalg.inv(self.poses_true[0])
        self.scale_factors = np.exp(self.rng.normal(0.0, scale_jitter, n_views))
        self.depth_noise = depth_noise
        self._cache = None

    # ---- rendering ----
    def _render(self, cam2world: np.ndarray):
        H, W, K = self.H, self.W, self.K
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        d = d_cam.reshape(-1, 3) @ cam2world[:3, :3].T
        o = cam2world[:3, 3]
        t_best = np.full(len(d), np.inf)
        ids = np.full(len(d), -1, dtype=int)

        def plane(n, off, bound=None):
            denom = d @ n
            t = np.where(np.abs(denom) > 1e-9, (off - o @ n) / np.where(denom == 0, 1, denom), np.inf)
            t = np.where(t > 1e-6, t, np.inf)
            if bound is not None:
                p = o + t[:, None] * d
                t = np.where(bound(p), t, np.inf)
            return t

        for t, code in ((plane(np.array([0, 0, 1.0]), 0.0), -2),
                        (plane(np.array([1.0, 0, 0]), -2.0, lambda p: (p[:, 2] >= 0) & (p[:, 2] <= 2.6)), -3),
                        (plane(np.array([0, 1.0, 0]), 2.0, lambda p: (p[:, 2] >= 0) & (p[:, 2] <= 2.6)), -4)):
            upd = t < t_best
            t_best[upd], ids[upd] = t[upd], code
        for k, b in enumerate(self.boxes):
            R = rot_z(b.yaw)
            ol, dl = (o - b.center) @ R, d @ R
            inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
            t1, t2 = (-b.size / 2 - ol) * inv, (b.size / 2 - ol) * inv
            tn, tf = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
            hit = (tn <= tf) & (tn > 1e-6) & (tn < t_best)
            t_best[hit], ids[hit] = tn[hit], k
        depth = np.where(np.isfinite(t_best), t_best, 0.0).reshape(H, W)
        return depth, ids.reshape(H, W)

    def render_all(self):
        if self._cache is None:
            self._cache = [self._render(T) for T in self.poses_true]
        return self._cache

    def frames(self) -> list[Frame]:
        out = []
        for i, (_, ids) in enumerate(self.render_all()):
            img = np.full((self.H, self.W, 3), 225, np.uint8)
            img[ids == -2] = (180, 170, 150)
            for k, b in enumerate(self.boxes):
                img[ids == k] = b.color
            out.append(Frame(index=i, image=img, source=f"synthetic#{i}"))
        return out

    # ---- ground truth in the frame the pipeline is expected to recover ----
    def true_dims(self, name: str) -> np.ndarray:
        b = next(b for b in self.boxes if b.name == name)
        L, W = sorted(b.size[:2], reverse=True)
        return np.array([L, W, b.size[2]])


class SyntheticGeometry:
    """Returns the true geometry with per-view scale error and noise."""

    def __init__(self, world: SyntheticWorld):
        self.world = world

    def infer(self, frames: list[Frame]) -> list[View]:
        w = self.world
        out = []
        for f, (depth, _), T, s in zip(frames, w.render_all(), w.poses_true, w.scale_factors):
            noisy = depth * s * (1 + w.rng.normal(0, w.depth_noise, depth.shape))
            valid = depth > 0
            out.append(View(index=f.index, image=f.image, depth=noisy.astype(np.float32), K=w.K.copy(),
                            cam2world=w.world_from_true @ T, conf=valid.astype(np.float32),
                            valid=valid, source=f.source))
        return out


class SyntheticSegmenter:
    """Perfect per-frame masks. No track ids, so association has to do the work."""

    def __init__(self, world: SyntheticWorld):
        self.world = world

    def segment(self, views: list[View]) -> list[list[Mask2D]]:
        out = []
        for view, (_, ids) in zip(views, self.world.render_all()):
            out.append([Mask2D(view.index, ids == k, b.name, 1.0)
                        for k, b in enumerate(self.world.boxes) if (ids == k).sum() > 0])
        return out


class SyntheticCompleter:
    """Returns the right shape in a deliberately wrong pose and scale, the way
    a generative model would."""

    def __init__(self, world: SyntheticWorld):
        self.world = world

    def complete(self, view: View, mask: Mask2D) -> Optional[CompletedShape]:
        b = next((b for b in self.world.boxes if b.name == mask.label), None)
        if b is None:
            return None
        m = trimesh.creation.box(extents=b.size)
        for _ in range(3):
            m = m.subdivide()
        R = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]) @ rot_z(np.deg2rad(17))
        verts = (np.asarray(m.vertices) @ R.T) * (1.0 / b.size.max()) + np.array([3.0, -1.0, 2.0])
        return CompletedShape(vertices=verts, faces=np.asarray(m.faces))
