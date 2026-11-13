# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 9: make the scene physically plausible.

Works on upright boxes (fast, robust, easy to explain):
  * every object gets a support: the floor, another object, or a wall;
  * small gaps and small sinks are closed by moving the object vertically;
  * small sideways overlaps are pushed apart, large ones are only reported,
    because a chair tucked under a table overlaps its box legitimately.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .geometry import footprint_mtv, make_transform, obb_footprint, points_in_footprint


def _footprint_grid(obb: dict, n: int = 5) -> np.ndarray:
    fp = obb_footprint(obb)
    a = np.linspace(0.1, 0.9, n)
    g = [fp[0] + s * (fp[1] - fp[0]) + t * (fp[3] - fp[0]) for s in a for t in a]
    return np.array(g)


def find_support(obj, scene, tol: float, any_distance: bool = False):
    """Returns (support_id, gap). gap > 0 means the object floats above its support."""
    grid = _footprint_grid(obj.obb)
    best, best_top = None, -np.inf
    for other in scene.objects.values():
        if other.id == obj.id or other.top > obj.obb["center"][2]:
            continue
        if not any_distance and other.top < obj.bottom - max(tol, 0.5 * obj.obb["size"][2]) - 0.3:
            continue  # far below: something else is in between, or the object floats
        if points_in_footprint(grid, other.obb, margin=0.01).mean() >= 0.3 and other.top > best_top:
            best, best_top = other.id, other.top
    if best is not None:
        return best, obj.bottom - best_top
    return "floor", obj.bottom


def _near_wall(obj, scene, slack: float = 0.12) -> bool:
    c = obj.obb["center"]
    half = 0.5 * float(np.linalg.norm(obj.obb["size"][:2]))
    for w in scene.layout.walls:
        if abs(float(w["normal"] @ c + w["d"])) <= half + slack:
            return True
    return False


def settle(scene, cfg: Config | None = None) -> None:
    """Assign supports, close vertical gaps, resolve small overlaps. Edits the scene in place."""
    cfg = cfg or Config()
    for o in scene.objects.values():
        o.flags = [f for f in o.flags if not f.startswith(("floating", "overlaps:"))]

    # bottom-up, so a supporter is already in place when its load is examined
    for obj in sorted(scene.objects.values(), key=lambda o: o.bottom):
        tol = max(cfg.support_tol, 0.1 * obj.obb["size"][2])
        was_removed_support = obj.support is not None and obj.support not in scene.objects \
            and obj.support not in ("floor", "wall")
        sup, gap = find_support(obj, scene, tol, any_distance=was_removed_support)
        # A measured floor is never perfectly flat, so an object can come out a
        # little below its support. Sunk into it by less than half its own
        # height, it rests on it just as one hovering within the tolerance does.
        sunk = -max(tol, 0.5 * obj.obb["size"][2])
        if sunk <= gap <= tol or was_removed_support:
            obj.apply(make_transform(t=np.array([0.0, 0.0, -gap])))
            obj.support = sup
        elif obj.support == "wall" or _near_wall(obj, scene):
            obj.support = "wall"
        else:
            obj.support = None
            obj.flags.append("floating")

    objs = list(scene.objects.values())
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if a.support == b.id or b.support == a.id:
                continue
            z_overlap = min(a.top, b.top) - max(a.bottom, b.bottom)
            if z_overlap <= 0.02:
                continue
            mtv = footprint_mtv(obb_footprint(a.obb), obb_footprint(b.obb))
            if mtv is None:
                continue
            small, big = (a, b) if np.prod(a.obb["size"]) <= np.prod(b.obb["size"]) else (b, a)
            if np.linalg.norm(mtv) <= cfg.max_push:
                shift = mtv if small is a else -mtv
                small.apply(make_transform(t=np.array([shift[0], shift[1], 0.0])))
            else:
                a.flags.append(f"overlaps:{b.id}")
                b.flags.append(f"overlaps:{a.id}")
