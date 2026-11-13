# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The full pipeline: frames in, Scene out. Read this file top to bottom to see
every step in order; each step lives in its own module."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from . import align, association, completion, fusion, ingest, physics, scale
from .config import Config
from .scene import Scene, SceneObject
from .types import Frame


class SceneLift:
    def __init__(self, geometry, segmenter, completer=None, config: Optional[Config] = None,
                 progress: Optional[Callable[[str, float], None]] = None):
        """geometry, segmenter, completer: objects implementing the interfaces in
        scenelift.backends.base. completer may be None (objects then get convex
        hulls). progress(stage_name, seconds) is called after every stage."""
        self.geometry = geometry
        self.segmenter = segmenter
        self.completer = completer
        self.cfg = config or Config()
        self.progress = progress

    @classmethod
    def default(cls, config: Optional[Config] = None, device: Optional[str] = None,
                sam3d_repo: Optional[str] = None, **kw) -> "SceneLift":
        """Build the reference stack: MapAnything + SAM 3 (+ SAM 3D Objects when
        sam3d_repo points at a checkout of that repository)."""
        from .backends.mapanything import MapAnythingBackend
        from .backends.sam3 import Sam3Backend

        cfg = config or Config()
        completer = None
        if sam3d_repo:
            from .backends.sam3d import Sam3DObjectsBackend
            completer = Sam3DObjectsBackend(sam3d_repo)
        return cls(MapAnythingBackend(device=device, conf_at_floor=cfg.min_conf),
                   Sam3Backend(cfg.prompts, device=device),
                   completer, cfg, **kw)

    def run(self, source, reference: Optional[tuple] = None) -> Scene:
        """source: a video file, a folder of photos, one photo, or a list of paths.
        reference: optional (object_id, "length"|"width"|"height", true_metres)
        to fix the absolute scale from one known size."""
        t0 = time.time()
        frames = ingest.load(source, self.cfg)
        scene = self.run_frames(frames, _t_ingest=time.time() - t0)
        if reference is not None:
            from .query import dimensions
            obj_id, axis, true = reference
            scene.calibrate(dimensions(scene, obj_id, self.cfg)[axis].value, float(true))
        return scene

    def run_frames(self, frames: list[Frame], _t_ingest: float = 0.0) -> Scene:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        timings = {"ingest": _t_ingest}

        def stage(name, fn):
            t = time.time()
            out = fn()
            timings[name] = time.time() - t
            if self.progress:
                self.progress(name, timings[name])
            return out

        # 2. metric geometry: depth, intrinsics and pose for every frame
        views = stage("geometry", lambda: self.geometry.infer(frames))
        # 3. make the views agree on scale, and measure how much they disagreed
        scale_info = stage("scale", lambda: scale.harmonise(views, cfg, rng))

        # 4. clean frame: Z up, floor at 0, walls square to the axes
        def do_align():
            T, floor_found = align.gravity_align(views, cfg, rng)
            align.apply_to_views(views, T)
            layout, T_yaw = align.estimate_layout(views, cfg, rng, floor_found)
            align.apply_to_views(views, T_yaw)
            return layout
        layout = stage("align", do_align)

        # 5. 2D instance masks in every view
        masks = stage("segment", lambda: self.segmenter.segment(views))
        # 6. link masks across views into 3D instances
        instances = stage("associate", lambda: association.associate(views, masks, cfg))

        # 7. fuse measured points per object, and the background
        def do_fuse():
            objs, used = {}, {}
            counts: dict = {}
            for inst in instances:
                fused = fusion.fuse_instance(inst, cfg)
                if fused is None:
                    continue
                label = inst.label
                counts[label] = counts.get(label, 0) + 1
                oid = f"{label}_{counts[label]}".replace(" ", "_")
                objs[oid] = SceneObject(id=oid, label=label, points=fused[0], colors=fused[1],
                                        n_views=inst.n_views)
                used[oid] = inst
            kept_masks = [[] for _ in views]
            pos = {v.index: i for i, v in enumerate(views)}
            for inst in used.values():
                for m in inst.observations:
                    kept_masks[pos[m.view_index]].append(m)
            bg = fusion.fuse_background(views, kept_masks, cfg)
            return objs, used, bg
        objects, inst_of, (bg_pts, bg_cols) = stage("fuse", do_fuse)

        # 8. complete each object and tag observed vs generated
        def do_complete():
            for oid, obj in objects.items():
                completion.complete_object(obj, inst_of[oid], views, self.completer, cfg, rng)
        stage("complete", do_complete)

        cameras = [{"K": v.K, "cam2world": v.cam2world, "size": v.size, "source": v.source,
                    "depth": v.depth if cfg.save_depth else None,
                    "image": v.image if cfg.save_frames else None} for v in views]
        scene = Scene(objects, layout, bg_pts, bg_cols, scale_info, cameras,
                      {"config": cfg.to_dict(), "timings": timings})
        # 9. supports, gravity, overlaps
        stage("physics", lambda: physics.settle(scene, cfg))
        scene.config["timings"] = timings
        return scene
