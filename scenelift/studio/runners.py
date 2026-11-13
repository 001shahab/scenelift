# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The two ways Studio makes a scene.

  * run_demo: the synthetic room with exact ground truth. Runs the whole
    geometry core on a laptop CPU in seconds, with no model weights.
  * run_reconstruction: real footage through the model backends
    (MapAnything + SAM 3, and SAM 3D Objects when configured).

Both drive scenelift.pipeline.SceneLift and report each stage to the job.
"""

from __future__ import annotations

import importlib.util
import threading
from functools import lru_cache
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np

from .. import ingest
from ..config import Config
from ..pipeline import SceneLift
from ..types import CompletedShape
from .jobs import PIPELINE_STAGES, Job
from .schemas import DemoRequest
from .store import SceneStore

DEMO_STAGES = PIPELINE_STAGES + ("save",)
RECONSTRUCT_STAGES = ("models",) + PIPELINE_STAGES + ("save",)


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def backend_status(sam3d_repo: Optional[str]) -> dict:
    """Which model backends can be imported. Nothing heavy is loaded here, and
    nothing is downloaded: whether the gated SAM 3 weights can be fetched is
    only known when the first reconstruction starts."""
    sam3d = bool(sam3d_repo) and (Path(sam3d_repo) / "notebook" / "inference.py").is_file()
    return {"torch": _installed("torch"), "mapanything": _installed("mapanything"),
            "sam3": _installed("torch") and _installed("transformers.models.sam3"),
            "sam3d": sam3d, "sam3d_repo": sam3d_repo}


class BackendPool:
    """Keeps the loaded models between jobs, so only the first reconstruction
    pays for loading the weights."""

    def __init__(self):
        self._lock = threading.Lock()
        self._geometry: dict = {}
        self._segmenter: dict = {}
        self._completer: dict = {}

    def pipeline(self, cfg: Config, device: Optional[str], sam3d_repo: Optional[str], progress) -> SceneLift:
        from ..backends.mapanything import MapAnythingBackend
        from ..backends.sam3 import Sam3Backend

        with self._lock:
            if device not in self._geometry:
                self._geometry[device] = MapAnythingBackend(device=device, conf_at_floor=cfg.min_conf)
            if device not in self._segmenter:
                self._segmenter[device] = Sam3Backend(cfg.prompts, device=device)
            segmenter = self._segmenter[device]
            segmenter.prompts = list(cfg.prompts)
            completer = None
            if sam3d_repo:
                if sam3d_repo not in self._completer:
                    from ..backends.sam3d import Sam3DObjectsBackend
                    self._completer[sam3d_repo] = Sam3DObjectsBackend(sam3d_repo)
                completer = self._completer[sam3d_repo]
            return SceneLift(self._geometry[device], segmenter, completer, cfg, progress=progress)


# --------------------------------------------------------------------------
# synthetic demo
# --------------------------------------------------------------------------
class WrongShapeCompleter:
    """Offers a sphere for every object. The completion checks of step 8 must
    reject it, fall back to the convex hull and set `completion_rejected`."""

    def complete(self, view, mask) -> Optional[CompletedShape]:
        import trimesh

        m = trimesh.creation.icosphere(subdivisions=3)
        return CompletedShape(np.asarray(m.vertices), np.asarray(m.faces))


def _true_gap(world, a: str, b: str, n: int = 40000) -> float:
    """Exact surface-to-surface gap between two boxes of the synthetic room."""
    import trimesh
    from scipy.spatial import cKDTree

    from ..geometry import rot_z

    clouds = []
    for k, name in enumerate((a, b)):
        box = next(x for x in world.boxes if x.name == name)
        mesh = trimesh.creation.box(extents=box.size)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, n, seed=k + 1)[0])
        clouds.append(pts @ rot_z(box.yaw).T + box.center)
    return float(cKDTree(clouds[1]).query(clouds[0])[0].min())


def demo_ground_truth(world, scene) -> list:
    """Tape-measure style ground truth for the synthetic room, in the JSON
    format read by scenelift.eval.evaluate."""
    gt = []
    for box in world.boxes:
        oid = f"{box.name}_1"
        if oid not in scene.objects:
            continue
        for axis, true in zip(("length", "width", "height"), world.true_dims(box.name)):
            gt.append({"type": "dimension", "object": oid, "axis": axis, "true_m": round(float(true), 4)})
    if "table_1" in scene.objects and "cabinet_1" in scene.objects:
        gt.append({"type": "distance", "a": "table_1", "b": "cabinet_1", "mode": "surface",
                   "true_m": round(_true_gap(world, "table", "cabinet"), 4)})
    if "box_1" in scene.objects:  # 0.25 x 0.20 x 0.25 m
        gt.append({"type": "fit", "object": "box_1", "space": [0.4, 0.4, 0.4], "true": True})
    if "cabinet_1" in scene.objects:  # 0.50 x 0.40 x 1.00 m
        gt.append({"type": "fit", "object": "cabinet_1", "space": [0.3, 0.3, 1.2], "true": False})
    return gt


def run_demo(job: Job, store: SceneStore, scene_id: str, req: DemoRequest) -> str:
    from ..backends.synthetic import (SyntheticCompleter, SyntheticGeometry, SyntheticSegmenter,
                                      SyntheticWorld)

    job.begin("ingest")
    t0 = time.time()
    world = SyntheticWorld(n_views=req.n_views, scale_jitter=req.scale_jitter,
                           depth_noise=req.depth_noise, seed=req.seed)
    frames = world.frames()
    t_ingest = time.time() - t0
    job.end("ingest", t_ingest)

    completer = {"correct": SyntheticCompleter(world), "wrong": WrongShapeCompleter(), "none": None}[req.completer]
    lift = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), completer, Config(),
                     progress=job.stage_done)
    job.begin("geometry")
    scene = lift.run_frames(frames, _t_ingest=t_ingest)

    job.begin("save")
    store.put(scene_id, scene, {
        "name": req.name, "source": "demo", "inputs": [f"synthetic room, {req.n_views} views"],
        "params": req.model_dump(), "ground_truth": demo_ground_truth(world, scene)})
    job.end("save")
    return scene_id


# --------------------------------------------------------------------------
# real footage
# --------------------------------------------------------------------------
def build_config(max_frames: Optional[int], prompts: Optional[list], overrides: Optional[dict]) -> Config:
    """A Config from the upload form. `overrides` may set any field of Config."""
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for key, value in (overrides or {}).items():
        if key not in known:
            raise ValueError(f"unknown config field '{key}'")
        current = getattr(cfg, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    if max_frames:
        cfg.max_frames = int(max_frames)
    if prompts:
        cfg.prompts = list(prompts)
    return cfg


def run_reconstruction(job: Job, store: SceneStore, pool: BackendPool, scene_id: str, source,
                       cfg: Config, meta: dict, device: Optional[str], sam3d_repo: Optional[str]) -> str:
    """source: one video path, or a folder of photos (see scenelift.ingest.load)."""
    job.begin("models")
    lift = pool.pipeline(cfg, device, sam3d_repo, job.stage_done)
    job.end("models")

    job.begin("ingest")
    t0 = time.time()
    frames = ingest.load(source, cfg)
    t_ingest = time.time() - t0
    job.end("ingest", t_ingest)

    job.begin("geometry")
    scene = lift.run_frames(frames, _t_ingest=t_ingest)

    job.begin("save")
    store.put(scene_id, scene, meta)
    job.end("save")
    return scene_id
