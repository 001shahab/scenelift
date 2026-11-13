# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Turn a Scene into what the browser needs.

Two payloads, so the heavy one is only fetched when the geometry changed:

  * the document: plain JSON with every number the inspector shows;
  * the geometry: vertex, face, colour and provenance arrays as base64 of
    little-endian typed arrays (float32, uint32, uint8), which the front end
    hands to WebGL without parsing.

Also the few geometric hints a viewer needs to draw an answer (the two end
points of a measured gap). They repeat the sampling of scenelift.query, so the
line drawn is the line measured.
"""

from __future__ import annotations

import base64
import math
from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .. import query
from ..config import Config
from ..geometry import obb_corners
from ..scene import Scene

MAX_OBJECT_POINTS = 40_000  # measured points sent per object
MAX_BACKGROUND_POINTS = 250_000


def jsonable(x):
    """Recursively convert numpy values to plain JSON. Non-finite floats become
    null, because strict JSON has no NaN or Infinity."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def config_of(scene: Scene) -> Config:
    """The Config the scene was built with, so answers use its own uncertainty
    constants. Unknown keys (a scene from another version) are ignored."""
    stored = (scene.config or {}).get("config") or {}
    known = {f.name for f in fields(Config)}
    values = {k: v for k, v in stored.items() if k in known}
    if "silhouette_scales" in values:
        values["silhouette_scales"] = tuple(values["silhouette_scales"])
    try:
        return Config(**values)
    except TypeError:
        return Config()


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------
def object_document(scene: Scene, obj, cfg: Config) -> dict:
    dims = query.dimensions(scene, obj.id, cfg)
    return {
        "id": obj.id,
        "label": obj.label,
        "completion": obj.completion,
        "support": obj.support,
        "flags": list(obj.flags),
        "n_views": int(obj.n_views),
        "observed_fraction": obj.observed_fraction,
        "fit": dict(obj.fit),
        "obb": {"center": obj.obb["center"], "size": obj.obb["size"], "yaw": float(obj.obb["yaw"])},
        "dimensions": {k: m.to_dict() for k, m in dims.items()},
        "n_points": int(len(obj.points)),
        "n_vertices": int(len(obj.mesh_vertices)) if obj.has_mesh else 0,
        "n_faces": int(len(obj.mesh_faces)) if obj.has_mesh else 0,
        "dependants": scene.dependants(obj.id),
    }


def scene_bounds(scene: Scene) -> dict:
    """Axis-aligned box around the objects and the room, for framing the view."""
    clouds = [obb_corners(o.obb) for o in scene.objects.values()]
    quads = [scene.layout.floor_quad] + [w["quad"] for w in scene.layout.walls]
    clouds += [np.asarray(q, dtype=float) for q in quads if np.abs(q).sum() > 0]
    if not clouds and len(scene.background_points):
        bg = scene.background_points
        clouds = [np.stack([np.percentile(bg, 1, axis=0), np.percentile(bg, 99, axis=0)])]
    if not clouds:
        return {"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]}
    pts = np.concatenate(clouds)
    return {"min": pts.min(axis=0), "max": pts.max(axis=0)}


def scene_document(scene: Scene, scene_id: str, meta: dict, revision: int) -> dict:
    cfg = config_of(scene)
    cameras = []
    for i, c in enumerate(scene.cameras):
        has_frame = c.get("image") is not None or bool(c.get("frame") and Path(c["frame"]).is_file())
        cameras.append({
            "index": i,
            "K": np.asarray(c["K"]),
            "cam2world": np.asarray(c["cam2world"]),
            "size": [int(c["size"][0]), int(c["size"][1])],  # (H, W)
            "source": c.get("source"),
            "has_frame": has_frame,
            "has_depth": c.get("depth") is not None,
        })
    return jsonable({
        "id": scene_id,
        "name": meta.get("name") or scene_id,
        "source": meta.get("source", "cli"),
        "created_at": meta.get("created_at"),
        "inputs": meta.get("inputs", []),
        "params": meta.get("params", {}),
        "ground_truth": meta.get("ground_truth"),
        "revision": revision,
        "units": "metres",
        "up_axis": "+Z",
        "objects": [object_document(scene, o, cfg) for o in scene.objects.values()],
        "layout": scene.layout.to_dict(),
        "scale": scene.scale.to_dict(),
        "cameras": cameras,
        "config": (scene.config or {}).get("config", {}),
        "timings": (scene.config or {}).get("timings", {}),
        "history": scene.history,
        "bounds": scene_bounds(scene),
    })


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _thin(points: np.ndarray, colors: np.ndarray, limit: int):
    """Evenly thin a cloud for display. The scene itself is never changed."""
    if len(points) <= limit:
        return points, colors
    keep = np.random.default_rng(0).choice(len(points), limit, replace=False)
    keep.sort()
    return points[keep], colors[keep]


def _cloud(points: np.ndarray, colors: np.ndarray, limit: int) -> dict:
    points, colors = _thin(np.asarray(points), np.asarray(colors), limit)
    return {"n_points": int(len(points)), "points": _b64(points, "<f4"), "colors": _b64(colors, "u1")}


def scene_geometry(scene: Scene, revision: int) -> dict:
    objects = {}
    for o in scene.objects.values():
        g = {"measured": _cloud(o.points, o.colors, MAX_OBJECT_POINTS), "mesh": None}
        if o.has_mesh and len(o.mesh_faces):
            n = len(o.mesh_vertices)
            colors = o.mesh_colors if o.mesh_colors is not None else np.full((n, 3), 160, np.uint8)
            observed = o.vertex_observed if o.vertex_observed is not None else np.ones(n, dtype=bool)
            g["mesh"] = {
                "n_vertices": int(n),
                "n_faces": int(len(o.mesh_faces)),
                "vertices": _b64(o.mesh_vertices, "<f4"),
                "faces": _b64(o.mesh_faces, "<u4"),
                "colors": _b64(np.asarray(colors)[:, :3], "u1"),
                "observed": _b64(np.asarray(observed), "u1"),
            }
        objects[o.id] = g
    return {
        "revision": revision,
        "objects": objects,
        "background": _cloud(scene.background_points, scene.background_colors, MAX_BACKGROUND_POINTS),
    }


# --------------------------------------------------------------------------
# hints for drawing answers
# --------------------------------------------------------------------------
def distance_segment(scene: Scene, a_id: str, b_id: str, mode: str = "surface") -> list:
    """The two end points of query.distance(scene, a, b, mode).

    Surface mode repeats the sampling of query.distance (same sample count, same
    seed, same 5th-smallest rule), so this is the pair the value came from.
    """
    a, b = scene[a_id], scene[b_id]
    if mode == "center":
        return [a.obb["center"], b.obb["center"]]
    pa, _ = a.surface_samples(6000)
    pb, _ = b.surface_samples(6000)
    dist, nn = cKDTree(pb).query(pa)
    k = min(4, len(dist) - 1)
    order = np.argpartition(dist, k)[: k + 1]
    i = order[np.argmax(dist[order])]
    return [pa[i], pb[nn[i]]]


def wall_segment(scene: Scene, obj_id: str) -> Optional[dict]:
    """Nearest wall of an object and a segment from the object straight to it,
    following query.distance_to_wall (1st percentile of the signed distance)."""
    pts, _ = scene[obj_id].surface_samples(4000)
    best = None
    for i, w in enumerate(scene.layout.walls):
        signed = pts @ w["normal"] + w["d"]
        d = float(np.percentile(signed, 1))
        if best is None or d < best[0]:
            best = (d, i, signed)
    if best is None:
        return None
    d, i, signed = best
    p = pts[int(np.argmin(np.abs(signed - d)))]
    n = np.asarray(scene.layout.walls[i]["normal"], dtype=float)
    return {"wall": i, "segment": [p, p - max(d, 0.0) * n]}
