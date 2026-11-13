# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 4: find the floor and the walls, then put the scene in a clean frame.

After this step: Z is up, the floor is z = 0, and the largest wall is parallel
to the X or Y axis. Everything downstream (upright boxes, support reasoning,
"will it fit") relies on this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .geometry import make_transform, ransac_plane, rot_z, rotation_between, transform_points, unproject
from .scale import usable_mask
from .types import View


@dataclass
class Layout:
    floor_found: bool = False
    floor_quad: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    walls: list = field(default_factory=list)  # each: dict(normal (3,), d, quad (4,3))

    def to_dict(self) -> dict:
        return {
            "floor_found": bool(self.floor_found),
            "floor_quad": self.floor_quad.tolist(),
            "walls": [{"normal": w["normal"].tolist(), "d": float(w["d"]),
                       "quad": w["quad"].tolist()} for w in self.walls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(bool(d["floor_found"]), np.array(d["floor_quad"]),
                   [{"normal": np.array(w["normal"]), "d": w["d"], "quad": np.array(w["quad"])}
                    for w in d["walls"]])


def sample_points(views: list[View], cfg: Config, rng: np.random.Generator,
                  per_view: int = 15000) -> np.ndarray:
    out = []
    for v in views:
        pts, _, _ = unproject(v.depth, v.K, v.cam2world, usable_mask(v, cfg))
        if len(pts) > per_view:
            pts = pts[rng.choice(len(pts), per_view, replace=False)]
        out.append(pts)
    return np.concatenate(out) if out else np.zeros((0, 3))


def up_prior(views: list[View]) -> np.ndarray:
    """People hold phones roughly upright, so 'up' is about minus the camera Y axis."""
    up = -np.mean([v.cam2world[:3, 1] for v in views], axis=0)
    n = np.linalg.norm(up)
    return up / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])


def gravity_align(views: list[View], cfg: Config, rng: np.random.Generator):
    """Detect the floor and return (T, floor_found); T maps the old world to Z-up, floor at 0.

    The floor is the plane whose normal is close to the up prior and that has
    almost nothing underneath it. The second condition is what separates it
    from table tops (floor points lie below them) and from the ceiling.
    """
    pts = sample_points(views, cfg, rng)
    up = up_prior(views)
    res = ransac_plane(pts, cfg.plane_thresh, cfg.plane_iters, rng, normal_prior=up,
                       max_angle_deg=cfg.floor_max_tilt_deg, below_penalty=4.0)
    found = res is not None and res[2].mean() > 0.03
    if found:
        n, d, _ = res
    else:  # no visible floor: trust the prior and put z = 0 at the lowest points
        n = up
        d = -float(np.percentile(pts @ n, 1.0)) if len(pts) else 0.0
    R = rotation_between(n, np.array([0.0, 0.0, 1.0]))
    T = make_transform(R, np.array([0.0, 0.0, d]))
    return T, bool(found)


def apply_to_views(views: list[View], T: np.ndarray) -> None:
    for v in views:
        v.cam2world = T @ v.cam2world


def estimate_layout(views: list[View], cfg: Config, rng: np.random.Generator,
                    floor_found: bool) -> tuple[Layout, np.ndarray]:
    """Find up to cfg.max_walls vertical planes and the yaw that squares the room.

    Call after gravity_align has been applied. Returns (layout, T_yaw); the
    caller applies T_yaw to the views. The layout is already expressed in the
    yaw-aligned frame.
    """
    pts = sample_points(views, cfg, rng)
    up = np.array([0.0, 0.0, 1.0])
    remaining = pts[pts[:, 2] > 0.15]
    n_total = max(len(remaining), 1)
    found = []
    for _ in range(cfg.max_walls):
        res = ransac_plane(remaining, cfg.plane_thresh * 1.5, cfg.plane_iters, rng,
                           normal_prior=up, max_angle_deg=8.0, perpendicular=True)
        if res is None:
            break
        n, d, inl = res
        if inl.sum() < cfg.wall_min_fraction * n_total:
            break
        # a wall is tall; this rejects the fronts of low furniture
        height = np.percentile(remaining[inl, 2], 98) - np.percentile(remaining[inl, 2], 2)
        if height > 1.0:
            n = np.array([n[0], n[1], 0.0])
            n /= np.linalg.norm(n)
            found.append((n, remaining[inl]))
        remaining = remaining[~inl]
        if len(remaining) < 200:
            break

    yaw = 0.0
    if found:
        n0 = max(found, key=lambda f: len(f[1]))[0]
        ang = np.arctan2(n0[1], n0[0])
        yaw = -(ang - np.round(ang / (np.pi / 2)) * (np.pi / 2))  # snap to the nearest axis
    T_yaw = make_transform(rot_z(yaw))

    layout = Layout(floor_found=floor_found)
    all_pts = transform_points(T_yaw, pts)
    if len(all_pts):
        lo = np.percentile(all_pts[:, :2], 0.5, axis=0) - 0.2
        hi = np.percentile(all_pts[:, :2], 99.5, axis=0) + 0.2
        layout.floor_quad = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0],
                                      [hi[0], hi[1], 0], [lo[0], hi[1], 0]], dtype=float)
    cam_centre = np.mean([transform_points(T_yaw, v.center[None])[0] for v in views], axis=0)
    for n, inl_pts in found:
        n = T_yaw[:3, :3] @ n
        p = transform_points(T_yaw, inl_pts)
        d = -float(np.median(p @ n))
        if cam_centre @ n + d < 0:  # normals point into the room
            n, d = -n, -d
        t = np.cross(up, n)
        a = p @ t
        a0, a1 = np.percentile(a, 1), np.percentile(a, 99)
        z1 = float(np.percentile(p[:, 2], 99))
        base = -d * n
        quad = np.array([base + a0 * t, base + a1 * t,
                         base + a1 * t + z1 * up, base + a0 * t + z1 * up])
        layout.walls.append({"normal": n, "d": d, "quad": quad})
    return layout, T_yaw
