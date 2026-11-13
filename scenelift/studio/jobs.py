# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Background jobs. A reconstruction takes seconds (synthetic) to minutes (real
footage), so it runs on a worker thread while the browser polls its progress.

One worker runs one job at a time: the model backends share a single GPU.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .store import now_iso

log = logging.getLogger("scenelift.studio")

# The stages of scenelift.pipeline.SceneLift, in order (docs/DESIGN.md, section 4).
PIPELINE_STAGES = ("ingest", "geometry", "scale", "align", "segment", "associate", "fuse",
                   "complete", "physics")


@dataclass
class Job:
    id: str
    kind: str  # "demo" | "reconstruct"
    name: str
    stage_names: tuple
    status: str = "queued"  # "queued" | "running" | "done" | "error"
    error: Optional[str] = None
    scene_id: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    finished_at: Optional[str] = None
    _stages: dict = field(default_factory=dict, repr=False)
    _started: dict = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self._stages = {n: {"name": n, "status": "pending", "seconds": None} for n in self.stage_names}

    # ---- progress, called from the worker thread ----
    def begin(self, stage: str) -> None:
        with self._lock:
            if stage in self._stages:
                self._stages[stage]["status"] = "running"
                self._started[stage] = time.time()

    def end(self, stage: str, seconds: Optional[float] = None) -> None:
        with self._lock:
            if stage in self._stages:
                if seconds is None:
                    seconds = time.time() - self._started.get(stage, time.time())
                self._stages[stage].update(status="done", seconds=round(float(seconds), 3))

    def stage_done(self, stage: str, seconds: float) -> None:
        """Progress hook for SceneLift: called after each pipeline stage, so the
        stage that follows is the one now running."""
        self.end(stage, seconds)
        names = list(self.stage_names)
        if stage in names and names.index(stage) + 1 < len(names):
            self.begin(names[names.index(stage) + 1])

    def fail(self, error: str) -> None:
        with self._lock:
            for s in self._stages.values():
                if s["status"] == "running":
                    s.update(status="error",
                             seconds=round(time.time() - self._started.get(s["name"], time.time()), 3))
        self.error = error
        self.status = "error"

    def to_dict(self) -> dict:
        with self._lock:
            stages = [dict(s) for s in self._stages.values()]
            now = time.time()
            for s in stages:
                if s["status"] == "running":
                    s["seconds"] = round(now - self._started.get(s["name"], now), 3)
        return {"id": self.id, "kind": self.kind, "name": self.name, "status": self.status,
                "error": self.error, "scene_id": self.scene_id, "created_at": self.created_at,
                "finished_at": self.finished_at, "stages": stages}


class JobManager:
    def __init__(self, keep: int = 50):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scenelift-job")

    def submit(self, kind: str, name: str, stages: tuple, fn: Callable[[Job], str]) -> Job:
        """Queue fn(job). It must return the id of the scene it stored."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, name=name, stage_names=tuple(stages))
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                self._jobs.popitem(last=False)
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], str]) -> None:
        job.status = "running"
        try:
            job.scene_id = fn(job)
            job.status = "done"
        except Exception as e:  # report every failure to the browser instead of dying silently
            log.exception("job %s (%s) failed", job.id, job.kind)
            job.fail(f"{type(e).__name__}: {e}")
        finally:
            job.finished_at = now_iso()

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"no job '{job_id}'")
            return self._jobs[job_id]

    def all(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
