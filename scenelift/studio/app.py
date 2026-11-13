# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The Studio web application: a REST API over the scenelift library, plus the
compiled React front end.

Every route is a thin wrapper. Reconstruction is scenelift.pipeline, edits are
methods of scenelift.scene.Scene, answers come from scenelift.query, and
selections from scenelift.select. Interactive API documentation is served at
/api/docs; docs/API.md lists every route.
"""

# No `from __future__ import annotations` here: FastAPI reads the route
# signatures at import time and needs the real types.
import json
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .. import __version__, query, select
from ..config import DEFAULT_PROMPTS, Config
from ..eval import evaluate
from ..ingest import IMAGE_EXT, VIDEO_EXT
from ..physics import find_support
from .jobs import PIPELINE_STAGES, JobManager
from .runners import (DEMO_STAGES, RECONSTRUCT_STAGES, BackendPool, backend_status, build_config,
                      run_demo, run_reconstruction)
from .schemas import (BoxRequest, CalibrateRequest, ClickRequest, DemoRequest, DistanceRequest,
                      EditRequest, EvaluateRequest, FitBetweenRequest, FitSpaceRequest, ObjectRequest,
                      PlacementsRequest, TextRequest, ToolCallRequest)
from .serialize import (config_of, distance_segment, jsonable, scene_document, scene_geometry,
                        wall_segment)
from .settings import StudioSettings
from .store import SceneEntry, SceneStore

STATIC_DIR = Path(__file__).parent / "static"

_MISSING_BUNDLE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SceneLift Studio</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:40rem;
margin:12vh auto;padding:0 1.5rem;color:#2C232E}code{background:#E6E7E8;padding:.15em .4em;
border-radius:4px}a{color:#2F7FD6}</style></head><body><h1>SceneLift Studio</h1>
<p>The server is running, but the compiled front end was not found in
<code>scenelift/studio/static</code>.</p><p>Build it once (needs Node.js 20 or newer):</p>
<pre><code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code></pre>
<p>The REST API works without it: <a href="/api/docs">/api/docs</a>.</p></body></html>"""


def create_app(settings: Optional[StudioSettings] = None) -> FastAPI:
    settings = settings or StudioSettings.from_env()
    store = SceneStore(settings.scenes_dir, settings.scene_cache)
    jobs = JobManager()
    pool = BackendPool()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    app = FastAPI(
        title="SceneLift Studio API",
        version=__version__,
        description="Reconstruct, inspect, edit and query metric 3D scenes. Lengths are metres. "
                    "The world is Z-up with the floor at z = 0. Every length comes with a 1-sigma "
                    "uncertainty and says whether it relies on generated geometry.",
        license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        contact={"name": "Prof. Shahab Anbarjafari, 3S Holding OÜ", "email": "shb@3sholding.com"},
        docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=5)
    app.state.settings, app.state.store, app.state.jobs = settings, store, jobs

    # ---- errors: the library raises KeyError for unknown ids, ValueError for bad input ----
    @app.exception_handler(KeyError)
    async def _key_error(_, exc: KeyError):
        return JSONResponse({"detail": str(exc.args[0]) if exc.args else "not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def document(entry: SceneEntry) -> dict:
        return scene_document(entry.scene, entry.id, store.meta(entry.id), store.revision(entry.id))

    def camera_index(entry: SceneEntry, index: int) -> int:
        if not 0 <= index < len(entry.scene.cameras):
            raise HTTPException(404, f"no camera {index}; the scene has {len(entry.scene.cameras)}")
        return index

    # ======================================================================
    # system
    # ======================================================================
    @app.get("/api/health", tags=["system"])
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/system", tags=["system"])
    def system():
        """Version, installed model backends and the defaults the front end needs."""
        backends = backend_status(settings.sam3d_repo)
        return {
            "name": "SceneLift Studio",
            "version": __version__,
            "scenes_dir": str(store.root.resolve()),
            "backends": backends,
            "can_reconstruct": bool(backends["mapanything"] and backends["sam3"]),
            "device": settings.device,
            "stages": list(PIPELINE_STAGES),
            "default_prompts": list(DEFAULT_PROMPTS),
            "default_max_frames": Config().max_frames,
            "accepted": {"images": sorted(IMAGE_EXT), "videos": sorted(VIDEO_EXT)},
            "max_upload_mb": settings.max_upload_mb,
            "license": "Apache-2.0",
            "copyright": "Copyright 2026 3S Holding OÜ",
        }

    # ======================================================================
    # jobs
    # ======================================================================
    @app.get("/api/jobs", tags=["jobs"])
    def list_jobs():
        return {"jobs": [j.to_dict() for j in jobs.all()]}

    @app.get("/api/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str):
        return jobs.get(job_id).to_dict()

    # ======================================================================
    # scenes
    # ======================================================================
    @app.get("/api/scenes", tags=["scenes"])
    def list_scenes():
        return {"scenes": store.summaries()}

    @app.post("/api/scenes/demo", status_code=202, tags=["scenes"])
    def create_demo(req: DemoRequest):
        """Reconstruct the synthetic room. Runs the complete geometry core on a
        CPU in seconds and stores exact ground truth with the scene."""
        scene_id = store.new_id(req.name)

        def task(job):
            try:
                return run_demo(job, store, scene_id, req)
            except Exception:
                store.discard(scene_id)
                raise

        return jobs.submit("demo", req.name, DEMO_STAGES, task).to_dict()

    @app.post("/api/scenes/reconstruct", status_code=202, tags=["scenes"])
    def create_reconstruction(
        files: list[UploadFile] = File(..., description="one video, or one or more photos"),
        name: str = Form("Untitled scene"),
        max_frames: Optional[int] = Form(None, ge=1, le=64),
        prompts: Optional[str] = Form(None, description="comma-separated object names to look for"),
        completion: bool = Form(True, description="use SAM 3D Objects when the server has it configured"),
        config: Optional[str] = Form(None, description="JSON object that overrides fields of Config"),
    ):
        """Reconstruct a scene from real footage with the model backends."""
        backends = backend_status(settings.sam3d_repo)
        if not (backends["mapanything"] and backends["sam3"]):
            state = ", ".join(f"{label} {'installed' if backends[key] else 'missing'}"
                              for key, label in (("mapanything", "MapAnything"), ("sam3", "SAM 3")))
            raise HTTPException(503, f"Real footage needs both model backends on this server ({state}). "
                                     "See docs/BACKENDS.md, or try the synthetic demo.")
        named = [(f, Path(f.filename or "").name) for f in files]
        videos = [x for x in named if Path(x[1]).suffix.lower() in VIDEO_EXT]
        images = [x for x in named if Path(x[1]).suffix.lower() in IMAGE_EXT]
        if len(videos) + len(images) != len(named):
            raise HTTPException(415, "Unsupported file type. Accepted: "
                                     + ", ".join(sorted(IMAGE_EXT | VIDEO_EXT)))
        if not ((len(videos) == 1 and not images) or (images and not videos)):
            raise HTTPException(422, "Upload either one video, or one or more photos.")
        overrides = json.loads(config) if config else None
        if overrides is not None and not isinstance(overrides, dict):
            raise HTTPException(422, "config must be a JSON object")
        prompt_list = [p.strip() for p in (prompts or "").split(",") if p.strip()]
        cfg = build_config(max_frames, prompt_list, overrides)

        upload_dir = store.root / ".uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True)
        budget = settings.max_upload_mb * 1024 * 1024
        try:
            ordered = sorted(named, key=lambda x: [int(t) if t.isdigit() else t.lower()
                                                   for t in re.split(r"(\d+)", x[1])])
            for i, (upload, filename) in enumerate(ordered):
                target = upload_dir / f"{i:04d}{Path(filename).suffix.lower()}"
                with target.open("wb") as out:
                    while chunk := upload.file.read(1 << 20):
                        budget -= len(chunk)
                        if budget < 0:
                            raise HTTPException(413, f"Upload is larger than {settings.max_upload_mb} MB.")
                        out.write(chunk)
        except Exception:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        source = next(upload_dir.iterdir()) if videos else upload_dir
        scene_id = store.new_id(name)
        sam3d_repo = settings.sam3d_repo if completion and backends["sam3d"] else None
        meta = {"name": name, "source": "upload", "inputs": [x[1] for x in ordered],
                "params": {"max_frames": cfg.max_frames, "prompts": cfg.prompts,
                           "completion": bool(sam3d_repo), "config": overrides or {}}}

        def task(job):
            try:
                return run_reconstruction(job, store, pool, scene_id, source, cfg, meta,
                                          settings.device, sam3d_repo)
            except Exception:
                store.discard(scene_id)
                raise
            finally:
                shutil.rmtree(upload_dir, ignore_errors=True)

        return jobs.submit("reconstruct", name, RECONSTRUCT_STAGES, task).to_dict()

    @app.get("/api/scenes/{scene_id}", tags=["scenes"])
    def get_scene(scene_id: str):
        """The scene document: objects with sizes and error bars, layout, scale,
        cameras, config, timings and edit history."""
        with store.use(scene_id) as entry:
            return document(entry)

    @app.get("/api/scenes/{scene_id}/geometry", tags=["scenes"])
    def get_geometry(scene_id: str):
        """Meshes, measured points and background as base64 typed arrays."""
        with store.use(scene_id) as entry:
            return scene_geometry(entry.scene, store.revision(scene_id))

    @app.delete("/api/scenes/{scene_id}", status_code=204, tags=["scenes"])
    def delete_scene(scene_id: str):
        store.delete(scene_id)
        return Response(status_code=204)

    @app.get("/api/scenes/{scene_id}/frames/{index}", tags=["scenes"])
    def get_frame(scene_id: str, index: int):
        """The keyframe of one camera as JPEG. For scenes saved without frames,
        a colour-mapped depth image stands in when depth was kept."""
        with store.use(scene_id) as entry:
            cam = entry.scene.cameras[camera_index(entry, index)]
            headers = {"Cache-Control": "private, max-age=3600"}
            if cam.get("frame") and Path(cam["frame"]).is_file():
                return FileResponse(cam["frame"], media_type="image/jpeg", headers=headers)
            image = entry.scene.frame_image(index)
            if image is not None:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if ok:
                    return Response(buf.tobytes(), media_type="image/jpeg", headers=headers)
            depth = cam.get("depth")
            if depth is not None:
                d = np.asarray(depth, dtype=np.float32)
                valid = np.isfinite(d) & (d > 0)
                if valid.any():
                    lo, hi = np.percentile(d[valid], [2, 98])
                    near = 1.0 - np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
                    vis = cv2.applyColorMap((255 * np.nan_to_num(near)).astype(np.uint8), cv2.COLORMAP_TURBO)
                    vis[~valid] = 0
                    ok, buf = cv2.imencode(".png", vis)
                    if ok:
                        return Response(buf.tobytes(), media_type="image/png", headers=headers)
        raise HTTPException(404, "this scene was saved without frames and without depth")

    @app.get("/api/scenes/{scene_id}/export", tags=["scenes"])
    def export_scene(scene_id: str,
                     fmt: Literal["glb", "obj", "ply", "points", "zip"] = Query("glb", alias="format"),
                     background: bool = Query(False, description="glb only: include the background points")):
        """Download the scene. glb, obj, ply: one named node per object plus floor
        and walls. points: every measured point as one PLY. zip: the scene folder."""
        tmp = Path(tempfile.mkdtemp(prefix="scenelift-export-"))
        try:
            with store.use(scene_id) as entry:
                if fmt == "zip":
                    out = Path(shutil.make_archive(str(tmp / scene_id), "zip", root_dir=store.path(scene_id)))
                elif fmt == "points":
                    out = tmp / f"{scene_id}_points.ply"
                    entry.scene.export_points(out)
                else:
                    work = tmp / "out"
                    work.mkdir()
                    entry.scene.export(work / f"{scene_id}.{fmt}", include_background=background and fmt == "glb")
                    written = sorted(work.iterdir())
                    # some formats write side files (materials); ship those as one archive
                    out = written[0] if len(written) == 1 else Path(
                        shutil.make_archive(str(tmp / f"{scene_id}_{fmt}"), "zip", root_dir=work))
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return FileResponse(out, filename=out.name, media_type="application/octet-stream",
                            background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))

    # ======================================================================
    # editing
    # ======================================================================
    @app.post("/api/scenes/{scene_id}/edit", tags=["edit"])
    def edit_scene(scene_id: str, req: EditRequest):
        """remove, translate, rotate, duplicate, drop or settle. Returns the
        updated scene document; reload the geometry when `revision` changed."""
        with store.use(scene_id) as entry:
            scene, extra = entry.scene, {}
            settle = req.settle
            if req.op != "settle" and not req.object:
                raise HTTPException(422, f"'{req.op}' needs an object")
            if req.op == "remove":
                scene.remove(req.object, settle=True if settle is None else settle)
                settle = False
            elif req.op == "translate":
                if req.delta is None:
                    raise HTTPException(422, "'translate' needs delta = [dx, dy, dz]")
                scene.translate(req.object, req.delta, carry=req.carry)
            elif req.op == "rotate":
                if req.yaw_deg is None:
                    raise HTTPException(422, "'rotate' needs yaw_deg")
                scene.rotate(req.object, req.yaw_deg, carry=req.carry)
            elif req.op == "duplicate":
                extra["new_id"] = scene.duplicate(req.object, req.offset)
            elif req.op == "drop":
                # lower the object onto whatever is underneath it, however far that is
                support, gap = find_support(scene[req.object], scene, 0.0, any_distance=True)
                scene.translate(req.object, (0.0, 0.0, -float(gap)), carry=req.carry)
                extra["support"] = support
                settle = True
            elif req.op == "settle":
                settle = True
            if settle:
                scene.settle()
            store.commit(entry)
            return {"scene": document(entry), **extra}

    @app.post("/api/scenes/{scene_id}/calibrate", tags=["edit"])
    def calibrate_scene(scene_id: str, req: CalibrateRequest):
        """Fix the absolute scale with one known length (Scene.calibrate)."""
        with store.use(scene_id) as entry:
            scene = entry.scene
            if req.measured_m is not None:
                measured = req.measured_m
            elif req.object:
                measured = query.dimensions(scene, req.object, config_of(scene))[req.axis].value
            else:
                raise HTTPException(422, "give an object and an axis, or measured_m")
            factor = scene.calibrate(measured, req.true_m)
            store.commit(entry)
            return {"factor": factor, "scene": document(entry)}

    # ======================================================================
    # spatial queries (scenelift.query)
    # ======================================================================
    @app.post("/api/scenes/{scene_id}/query/dimensions", tags=["query"])
    def query_dimensions(scene_id: str, req: ObjectRequest):
        with store.use(scene_id) as entry:
            dims = query.dimensions(entry.scene, req.object, config_of(entry.scene))
            return {k: m.to_dict() for k, m in dims.items()}

    @app.post("/api/scenes/{scene_id}/query/distance", tags=["query"])
    def query_distance(scene_id: str, req: DistanceRequest):
        """Gap between two objects, with the two end points for drawing it."""
        if req.a == req.b:
            raise HTTPException(422, "choose two different objects")
        with store.use(scene_id) as entry:
            scene = entry.scene
            m = query.distance(scene, req.a, req.b, req.mode, config_of(scene))
            return jsonable({**m.to_dict(), "segment": distance_segment(scene, req.a, req.b, req.mode)})

    @app.post("/api/scenes/{scene_id}/query/distance-to-wall", tags=["query"])
    def query_distance_to_wall(scene_id: str, req: ObjectRequest):
        """Gap to the nearest wall, or null when the scene has no walls."""
        with store.use(scene_id) as entry:
            scene = entry.scene
            m = query.distance_to_wall(scene, req.object, config_of(scene))
            if m is None:
                return None
            return jsonable({**m.to_dict(), **(wall_segment(scene, req.object) or {})})

    @app.post("/api/scenes/{scene_id}/query/fits-in-space", tags=["query"])
    def query_fits_in_space(scene_id: str, req: FitSpaceRequest):
        with store.use(scene_id) as entry:
            scene = entry.scene
            return query.fits_in_space(scene, req.object, req.space, req.allow_rotation, req.clearance,
                                       config_of(scene)).to_dict()

    @app.post("/api/scenes/{scene_id}/query/fits-between", tags=["query"])
    def query_fits_between(scene_id: str, req: FitBetweenRequest):
        if len({req.object, req.a, req.b}) < 3:
            raise HTTPException(422, "choose three different objects")
        with store.use(scene_id) as entry:
            scene, cfg = entry.scene, config_of(entry.scene)
            res = query.fits_between(scene, req.object, req.a, req.b, req.clearance, cfg)
            gap = query.distance(scene, req.a, req.b, "surface", cfg)
            return jsonable({**res.to_dict(), "gap": gap.to_dict(),
                             "segment": distance_segment(scene, req.a, req.b)})

    @app.post("/api/scenes/{scene_id}/query/placements", tags=["query"])
    def query_placements(scene_id: str, req: PlacementsRequest):
        """Free, observed floor positions where the object could stand."""
        if req.object is None and req.footprint is None:
            raise HTTPException(422, "give an object or a footprint")
        with store.use(scene_id) as entry:
            scene = entry.scene
            found = query.find_placements(scene, req.object, req.footprint, req.height, req.clearance,
                                          max_results=req.max_results, cfg=config_of(scene))
            if req.object is not None:
                length, width, height = (float(x) for x in scene[req.object].obb["size"])
            else:
                (length, width), height = req.footprint, req.height or 1.0
            return jsonable({"placements": found, "footprint": [length, width], "height": height})

    # ======================================================================
    # selection (scenelift.select)
    # ======================================================================
    @app.post("/api/scenes/{scene_id}/select/click", tags=["select"])
    def select_click(scene_id: str, req: ClickRequest):
        """Object under pixel (u, v) of one keyframe; null for background."""
        with store.use(scene_id) as entry:
            return {"object": select.by_click(entry.scene, camera_index(entry, req.camera), req.u, req.v)}

    @app.post("/api/scenes/{scene_id}/select/box", tags=["select"])
    def select_box(scene_id: str, req: BoxRequest):
        """Objects inside a box drawn on one keyframe, largest share first."""
        u0, v0, u1, v1 = req.box
        box = (min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1))
        with store.use(scene_id) as entry:
            return {"objects": select.by_box(entry.scene, camera_index(entry, req.camera), box)}

    @app.post("/api/scenes/{scene_id}/select/text", tags=["select"])
    def select_text(scene_id: str, req: TextRequest):
        with store.use(scene_id) as entry:
            return {"objects": select.by_text(entry.scene, req.text)}

    # ======================================================================
    # tools for language models, evaluation
    # ======================================================================
    @app.get("/api/tools", tags=["tools"])
    def list_tools():
        """query.TOOLS: JSON tool definitions for function calling."""
        return {"tools": query.TOOLS}

    @app.post("/api/scenes/{scene_id}/tools/call", tags=["tools"])
    def call_tool(scene_id: str, req: ToolCallRequest):
        """Run one tool exactly as a language model would call it."""
        spec = next((t for t in query.TOOLS if t["name"] == req.name), None)
        if spec is None:
            raise HTTPException(404, f"unknown tool '{req.name}'")
        missing = [k for k in spec["parameters"].get("required", []) if k not in req.arguments]
        if missing:
            raise HTTPException(422, f"missing argument(s): {', '.join(missing)}")
        with store.use(scene_id) as entry:
            return jsonable({"name": req.name, "arguments": req.arguments,
                             "result": query.call_tool(entry.scene, req.name, req.arguments)})

    @app.post("/api/scenes/{scene_id}/evaluate", tags=["tools"])
    def evaluate_scene(scene_id: str, req: EvaluateRequest):
        """Accuracy and calibration against tape-measure ground truth (scenelift.eval)."""
        for i, g in enumerate(req.ground_truth):
            need = {"dimension": ("object", "axis", "true_m"), "distance": ("a", "b", "true_m"),
                    "fit": ("object", "space", "true")}.get(g.get("type"))
            if need is None:
                raise HTTPException(422, f"entry {i}: type must be dimension, distance or fit")
            lacking = [k for k in need if k not in g]
            if lacking:
                raise HTTPException(422, f"entry {i}: missing {', '.join(lacking)}")
        with store.use(scene_id) as entry:
            return jsonable(evaluate(entry.scene, req.ground_truth))

    # ======================================================================
    # front end
    # ======================================================================
    if (STATIC_DIR / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def front_end(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "no such API route")
        root = STATIC_DIR.resolve()
        candidate = (root / path).resolve()
        if path and candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse(_MISSING_BUNDLE, status_code=503)

    return app
