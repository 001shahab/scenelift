# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 3: make metric scale consistent across views and attach an honest error bar.

Feed-forward geometry models return metric depth, but the scale of each view
can drift by a few percent. Two things are done about it:

1. Harmonise. For view i, every point seen by the OTHER views is projected
   into view i and compared with view i's own depth. The median log ratio is
   the scale error of view i relative to the rest. Dividing it out makes the
   views agree.

2. Quantify. The spread of those per-view factors measures how unstable the
   backbone's scale is on this scene. It is combined with a prior on the
   backbone's absolute scale error, because a bias shared by all views cannot
   be seen from the images alone. A user-supplied reference length replaces
   the prior (see Scene.calibrate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config
from .geometry import depth_edge_mask, project, unproject
from .types import View


@dataclass
class ScaleInfo:
    per_view: list = field(default_factory=list)  # multiplicative correction applied to each view
    sigma_consistency: float = 0.0  # robust std of log scale across views
    sigma_backbone: float = 0.04  # prior on the shared absolute scale error
    calibrated: bool = False  # True once a real-world reference was applied
    sigma_reference: float = 0.0  # relative error of that reference
    global_factor: float = 1.0  # factor applied by calibration
    n_views: int = 1

    @property
    def sigma_rel(self) -> float:
        """Relative 1-sigma error of any length measured in the scene."""
        if self.calibrated:
            return float(self.sigma_reference)
        # after harmonisation the shared scale is the median over the views, so
        # the view-to-view spread enters as the standard error of that median
        shared = 1.25 * self.sigma_consistency / np.sqrt(max(self.n_views, 1))
        return float(np.hypot(self.sigma_backbone, shared))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sigma_rel"] = self.sigma_rel
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScaleInfo":
        d = {k: v for k, v in d.items() if k != "sigma_rel"}
        return cls(**d)


def usable_mask(view: View, cfg: Config) -> np.ndarray:
    """Pixels that are trustworthy enough to become 3D points."""
    return (view.valid & (view.conf >= cfg.min_conf) & (view.depth > 0.05)
            & (view.depth < cfg.max_depth) & depth_edge_mask(view.depth, cfg.depth_edge_rel))


def _sample_cloud(view: View, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    pts, _, _ = unproject(view.depth, view.K, view.cam2world, usable_mask(view, cfg))
    if len(pts) > cfg.scale_points_per_view:
        pts = pts[rng.choice(len(pts), cfg.scale_points_per_view, replace=False)]
    return pts


def _log_scale_errors(views: list[View], cfg: Config, rng: np.random.Generator) -> np.ndarray:
    clouds = [_sample_cloud(v, cfg, rng) for v in views]
    out = np.full(len(views), np.nan)
    for i, v in enumerate(views):
        others = [c for j, c in enumerate(clouds) if j != i and len(c)]
        if not others:
            continue
        P = np.concatenate(others)
        u, vv, z = project(P, v.K, v.world2cam)
        H, W = v.size
        ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, z = ui[ok], vi[ok], z[ok]
        good = usable_mask(v, cfg)[vi, ui]
        d = v.depth[vi, ui]
        r = np.log(z[good] / d[good])
        r = r[np.abs(r) < cfg.scale_gate]  # larger ratios are occlusions, not scale
        if len(r) >= cfg.scale_min_matches:
            out[i] = np.median(r)
    return out


def harmonise(views: list[View], cfg: Config, rng: np.random.Generator) -> ScaleInfo:
    """Rescale each view's depth in place so all views agree. Returns ScaleInfo."""
    info = ScaleInfo(per_view=[1.0] * len(views), sigma_backbone=cfg.backbone_scale_sigma,
                     n_views=len(views))
    if len(views) < 2:
        return info
    total = np.zeros(len(views))
    first = None
    for _ in range(2):  # two rounds are enough in practice
        ls = _log_scale_errors(views, cfg, rng)
        if first is None:
            first = ls.copy()
        fin = np.isfinite(ls)
        if fin.sum() < 2:
            break
        ls = np.where(fin, ls - np.median(ls[fin]), 0.0)  # fix the gauge: no global drift
        for v, l in zip(views, ls):
            v.depth = (v.depth * np.exp(l)).astype(np.float32)
        total += ls
    info.per_view = [float(np.exp(t)) for t in total]
    fin = np.isfinite(first)
    if fin.sum() >= 2:
        f = first[fin]
        info.sigma_consistency = float(1.4826 * np.median(np.abs(f - np.median(f))))
    return info
