# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Spatial questions answered by computing on the scene, with error bars.

Every answer is a Measurement or a FitResult that carries:
  * the value in metres,
  * a 1-sigma uncertainty,
  * a flag saying whether the answer depends on generated (unseen) geometry.

Uncertainty model (deliberately simple and stated openly):
  sigma^2 = (value * sigma_rel)^2          scale error, grows with length
          + 2 * sigma_local^2              surface noise at the two end points
          + sum (k * h)^2                  for each end point on generated
                                           geometry, h = how far that point is
                                           from the nearest measured point
The constants live in Config and are meant to be calibrated on the benchmark
(see eval.py); they are not claimed to be exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

from .config import Config


@dataclass
class Measurement:
    value: float
    sigma: float
    relies_on_generated: bool = False
    what: str = ""

    @property
    def interval95(self) -> tuple:
        return (self.value - 1.96 * self.sigma, self.value + 1.96 * self.sigma)

    def to_dict(self) -> dict:
        lo, hi = self.interval95
        return {"what": self.what, "value_m": round(self.value, 4), "sigma_m": round(self.sigma, 4),
                "interval95_m": [round(lo, 4), round(hi, 4)],
                "relies_on_generated": bool(self.relies_on_generated)}

    def __str__(self) -> str:
        lo, hi = self.interval95
        note = " (partly based on unseen geometry)" if self.relies_on_generated else ""
        return f"{self.what}: {self.value:.3f} m +/- {self.sigma:.3f} (95%: {lo:.3f} to {hi:.3f}){note}"


@dataclass
class FitResult:
    probability: float
    verdict: str  # "yes" | "no" | "uncertain"
    margins: dict = field(default_factory=dict)  # axis -> Measurement of spare room
    orientation: str = ""
    relies_on_generated: bool = False

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "probability": round(self.probability, 4),
                "orientation": self.orientation, "relies_on_generated": bool(self.relies_on_generated),
                "margins": {k: m.to_dict() for k, m in self.margins.items()}}

    def __str__(self) -> str:
        parts = ", ".join(f"{k} spare {m.value * 100:.1f} cm +/- {m.sigma * 100:.1f}"
                          for k, m in self.margins.items())
        return f"fits: {self.verdict} (p = {self.probability:.2f}; {parts})"


def _cfg(scene, cfg: Optional[Config]) -> Config:
    return cfg or Config()


def _verdict(p: float, cfg: Config) -> str:
    return "yes" if p >= cfg.fit_yes else "no" if p <= cfg.fit_no else "uncertain"


def _observed_extent(obj) -> np.ndarray:
    """Size of the MEASURED part of the object along its box axes (L, W, H)."""
    c, s = np.cos(obj.obb["yaw"]), np.sin(obj.obb["yaw"])
    p = obj.points
    a = p[:, :2] @ np.array([c, s])
    b = p[:, :2] @ np.array([-s, c])
    return np.array([np.ptp(a), np.ptp(b), np.ptp(p[:, 2])])


def dimensions(scene, obj_id: str, cfg: Optional[Config] = None) -> dict:
    """Length, width (horizontal, length >= width) and height of an object."""
    cfg = _cfg(scene, cfg)
    obj = scene[obj_id]
    size = obj.obb["size"]
    seen = np.minimum(_observed_extent(obj), size)
    out = {}
    for name, full, obs in zip(("length", "width", "height"), size, seen):
        unseen = max(float(full - obs), 0.0)
        sigma = np.sqrt((full * scene.scale.sigma_rel) ** 2 + 2 * cfg.sigma_local ** 2
                        + (cfg.generated_sigma_frac * unseen) ** 2)
        out[name] = Measurement(float(full), float(sigma), unseen > 0.1 * full, f"{obj_id} {name}")
    return out


def distance(scene, a_id: str, b_id: str, mode: str = "surface",
             cfg: Optional[Config] = None) -> Measurement:
    """Distance between two objects.

    mode="surface": the gap between the closest points of the two surfaces.
    mode="center":  between the centres of their boxes.
    """
    cfg = _cfg(scene, cfg)
    a, b = scene[a_id], scene[b_id]
    sr = scene.scale.sigma_rel
    if mode == "center":
        d = float(np.linalg.norm(a.obb["center"] - b.obb["center"]))
        sigma = np.sqrt((d * sr) ** 2 + 2 * (2 * cfg.sigma_local) ** 2)
        gen = a.observed_fraction < 0.5 or b.observed_fraction < 0.5
        return Measurement(d, float(sigma), gen, f"centre distance {a_id} to {b_id}")
    pa, oa = a.surface_samples(6000)
    pb, ob = b.surface_samples(6000)
    dist, nn = cKDTree(pb).query(pa)
    k = min(4, len(dist) - 1)  # 5th smallest: robust to one stray sample
    order = np.argpartition(dist, k)[: k + 1]
    i = order[np.argmax(dist[order])]
    d = float(dist[i])
    var = (d * sr) ** 2 + 2 * cfg.sigma_local ** 2
    gen = False
    for obj, pt, is_obs in ((a, pa[i], oa[i]), (b, pb[nn[i]], ob[nn[i]])):
        if not is_obs:
            gen = True
            h = float(cKDTree(obj.points).query(pt)[0])
            var += (cfg.generated_sigma_frac * h) ** 2
    return Measurement(d, float(np.sqrt(var)), gen, f"gap between {a_id} and {b_id}")


def distance_to_wall(scene, obj_id: str, cfg: Optional[Config] = None) -> Optional[Measurement]:
    """Gap between an object and the nearest wall, or None when no wall was found."""
    cfg = _cfg(scene, cfg)
    obj = scene[obj_id]
    pts, _ = obj.surface_samples(4000)
    best = None
    for i, w in enumerate(scene.layout.walls):
        d = float(np.percentile(pts @ w["normal"] + w["d"], 1))
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return None
    d = max(best[0], 0.0)
    sigma = np.sqrt((d * scene.scale.sigma_rel) ** 2 + 2 * cfg.sigma_local ** 2)
    return Measurement(d, float(sigma), obj.observed_fraction < 0.5, f"gap between {obj_id} and wall_{best[1]}")


def fits_in_space(scene, obj_id: str, space, allow_rotation: bool = True, clearance: float = 0.0,
                  cfg: Optional[Config] = None) -> FitResult:
    """Will the object fit in a space of (width, depth, height) metres?

    The space is taken as exactly known (a tape measure, a product sheet), so
    the full scale error of the scene applies to the object's size.
    """
    cfg = _cfg(scene, cfg)
    dims = dimensions(scene, obj_id, cfg)
    L, W, H = dims["length"], dims["width"], dims["height"]
    sw, sd, sh = (float(x) for x in space)
    options = [("as scanned", (L, sw), (W, sd))]
    if allow_rotation:
        options += [("turned 90 degrees", (L, sd), (W, sw))]
    best = None
    for name, (m1, s1), (m2, s2) in options:
        margins = {
            "length": Measurement(s1 - m1.value - clearance, m1.sigma, m1.relies_on_generated, "spare room"),
            "width": Measurement(s2 - m2.value - clearance, m2.sigma, m2.relies_on_generated, "spare room"),
            "height": Measurement(sh - H.value - clearance, H.sigma, H.relies_on_generated, "spare room"),
        }
        p = float(np.prod([norm.cdf(m.value / max(m.sigma, 1e-9)) for m in margins.values()]))
        if best is None or p > best.probability:
            best = FitResult(p, _verdict(p, cfg), margins, name,
                             any(m.relies_on_generated for m in margins.values()))
    return best


def fits_between(scene, obj_id: str, a_id: str, b_id: str, clearance: float = 0.0,
                 cfg: Optional[Config] = None) -> FitResult:
    """Will the object fit in the gap between two other objects of the same scene?

    Both lengths come from the same reconstruction, so a scale error stretches
    them together and only acts on their difference. That makes this answer
    tighter than comparing against an outside number.
    """
    cfg = _cfg(scene, cfg)
    gap = distance(scene, a_id, b_id, "surface", cfg)
    dims = dimensions(scene, obj_id, cfg)
    sr = scene.scale.sigma_rel
    best = None
    for name in ("width", "length"):
        m = dims[name]
        margin = gap.value - m.value - clearance
        local_gap = max(gap.sigma ** 2 - (gap.value * sr) ** 2, 0.0)
        local_obj = max(m.sigma ** 2 - (m.value * sr) ** 2, 0.0)
        sigma = float(np.sqrt((margin * sr) ** 2 + local_gap + local_obj))
        p = float(norm.cdf(margin / max(sigma, 1e-9)))
        gen = gap.relies_on_generated or m.relies_on_generated
        res = FitResult(p, _verdict(p, cfg), {name: Measurement(margin, sigma, gen, "spare room")},
                        f"{name} across the gap", gen)
        if best is None or p > best.probability:
            best = res
    return best


def find_placements(scene, obj_id: Optional[str] = None, footprint=None, height: Optional[float] = None,
                    clearance: float = 0.02, cell: float = 0.02, max_results: int = 5,
                    cfg: Optional[Config] = None) -> list:
    """Where on the floor could this object (or a footprint of (length, width)) stand?

    Builds a top-down grid. A cell is usable only if the floor there was
    actually observed and nothing occupies it below the object's height.
    Unobserved floor counts as unknown, not as free. Returns up to max_results
    dicts(center_xy, yaw_deg, spare_m), best first; empty when nothing fits.
    """
    cfg = _cfg(scene, cfg)
    if obj_id is not None:
        L, W, Hh = scene[obj_id].obb["size"]
    else:
        L, W = footprint
        Hh = height if height is not None else 1.0
    q = scene.layout.floor_quad
    lo, hi = q[:, :2].min(axis=0), q[:, :2].max(axis=0)
    shape = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
    if shape.max() > 4000:
        raise ValueError("scene too large for the placement grid; increase `cell`")

    def cells(xy):
        ij = np.floor((xy - lo) / cell).astype(int)
        ok = (ij >= 0).all(axis=1) & (ij < shape).all(axis=1)
        return ij[ok]

    floor_seen = np.zeros(shape, np.uint8)
    blocked = np.zeros(shape, np.uint8)
    clouds = [scene.background_points] + [o.surface_samples(20000)[0] for o in scene.objects.values()
                                          if o.id != obj_id]
    for p in clouds:
        if len(p) == 0:
            continue
        f = cells(p[np.abs(p[:, 2]) < 0.04, :2])
        floor_seen[f[:, 0], f[:, 1]] = 1
        b = cells(p[(p[:, 2] > 0.05) & (p[:, 2] < Hh + 0.05), :2])
        blocked[b[:, 0], b[:, 1]] = 1
    gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    centres = np.stack([gx, gy], axis=-1).reshape(-1, 2) * cell + lo + cell / 2
    for w in scene.layout.walls:
        behind = (centres @ w["normal"][:2] + w["d"]) < 0
        blocked.reshape(-1)[behind] = 1
    k = max(3, int(round(0.10 / cell)) | 1)
    floor_seen = cv2.morphologyEx(floor_seen, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    free = ((floor_seen > 0) & (blocked == 0)).astype(np.uint8)

    results = []
    for yaw_deg, (sx, sy) in ((0.0, (L, W)), (90.0, (W, L))):
        kx = max(1, int(np.ceil((sx + 2 * clearance) / cell)))
        ky = max(1, int(np.ceil((sy + 2 * clearance) / cell)))
        ok = cv2.erode(free, np.ones((kx, ky), np.uint8), borderValue=0)
        if not ok.any():
            continue
        spare = cv2.distanceTransform(ok, cv2.DIST_L2, 3) * cell
        n_lab, lab = cv2.connectedComponents(ok)
        for l in range(1, n_lab):
            region = np.where(lab == l, spare, -1)
            i, j = np.unravel_index(np.argmax(region), region.shape)
            results.append({"center_xy": [float(lo[0] + (i + 0.5) * cell), float(lo[1] + (j + 0.5) * cell)],
                            "yaw_deg": yaw_deg, "spare_m": float(spare[i, j])})
    results.sort(key=lambda r: -r["spare_m"])
    return results[:max_results]


# --------------------------------------------------------------------------
# tool interface for language models
# --------------------------------------------------------------------------
# A language model should never guess a distance. Give it these tools and let
# it call them; the numbers then come from the reconstruction.
TOOLS = [
    {"name": "list_objects", "description": "List every object with size, position and support.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_dimensions", "description": "Length, width and height of one object in metres.",
     "parameters": {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}},
    {"name": "measure_distance", "description": "Distance between two objects in metres.",
     "parameters": {"type": "object", "properties": {
         "a": {"type": "string"}, "b": {"type": "string"},
         "mode": {"type": "string", "enum": ["surface", "center"]}}, "required": ["a", "b"]}},
    {"name": "check_fit_in_space", "description": "Does an object fit in a space of width, depth, height (metres)?",
     "parameters": {"type": "object", "properties": {
         "object": {"type": "string"}, "width": {"type": "number"}, "depth": {"type": "number"},
         "height": {"type": "number"}}, "required": ["object", "width", "depth", "height"]}},
    {"name": "check_fit_between", "description": "Does an object fit in the gap between two other objects?",
     "parameters": {"type": "object", "properties": {
         "object": {"type": "string"}, "a": {"type": "string"}, "b": {"type": "string"}},
         "required": ["object", "a", "b"]}},
    {"name": "find_placements", "description": "Free floor positions where an object could stand.",
     "parameters": {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}},
]


def _resolve(scene, name: str) -> str:
    if name in scene.objects:
        return name
    hits = scene.find(name)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no object matches '{name}'. Available: {', '.join(scene.ids)}")
    raise KeyError(f"'{name}' is ambiguous: {', '.join(hits)}")


def call_tool(scene, name: str, args: dict) -> dict:
    """Run one of TOOLS and return a JSON-serialisable result."""
    if name == "list_objects":
        return {"objects": scene.summary(), "scale_sigma_rel": scene.scale.sigma_rel}
    if name == "get_dimensions":
        o = _resolve(scene, args["object"])
        return {k: m.to_dict() for k, m in dimensions(scene, o).items()}
    if name == "measure_distance":
        return distance(scene, _resolve(scene, args["a"]), _resolve(scene, args["b"]),
                        args.get("mode", "surface")).to_dict()
    if name == "check_fit_in_space":
        return fits_in_space(scene, _resolve(scene, args["object"]),
                             (args["width"], args["depth"], args["height"])).to_dict()
    if name == "check_fit_between":
        return fits_between(scene, _resolve(scene, args["object"]), _resolve(scene, args["a"]),
                            _resolve(scene, args["b"])).to_dict()
    if name == "find_placements":
        return {"placements": find_placements(scene, _resolve(scene, args["object"]))}
    raise KeyError(f"unknown tool '{name}'")
