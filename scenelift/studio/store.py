# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The scene library: one folder per scene inside a root folder.

Each scene folder is an ordinary scenelift scene (``scene.json`` plus arrays,
see docs/SCENE_FORMAT.md), so anything written by ``scenelift run -o
scenes/<name>`` shows up in Studio, and anything made in Studio can be used
from Python or the command line. Studio adds one small file, ``studio.json``,
with the display name and how the scene was made.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from ..scene import Scene

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
META_FILE = "studio.json"


class SceneNotFound(KeyError):
    """Raised for a scene id that is malformed or not in the library."""


@dataclass
class SceneEntry:
    """A scene held in memory. Take ``lock`` (via SceneStore.use) before touching it."""

    id: str
    scene: Scene
    lock: threading.RLock = field(default_factory=threading.RLock)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:40].strip("-")
    return slug or "scene"


class SceneStore:
    def __init__(self, root: Path, cache_size: int = 6):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, SceneEntry] = OrderedDict()
        self._revisions: dict[str, int] = {}
        self._cache_size = max(1, cache_size)
        self._lock = threading.RLock()

    # ---- paths and ids ----
    def path(self, scene_id: str) -> Path:
        """Folder of a scene. The id is validated, so it can never leave the root."""
        if not _ID_RE.match(scene_id or ""):
            raise SceneNotFound(f"invalid scene id '{scene_id}'")
        return self.root / scene_id

    def exists(self, scene_id: str) -> bool:
        try:
            return (self.path(scene_id) / "scene.json").is_file()
        except SceneNotFound:
            return False

    def new_id(self, name: str) -> str:
        """A free folder name derived from a display name."""
        with self._lock:
            base = slugify(name)
            candidate, n = base, 2
            while (self.root / candidate).exists():
                candidate = f"{base}-{n}"
                n += 1
            (self.root / candidate).mkdir(parents=True)  # reserve it
            return candidate

    # ---- metadata ----
    def meta(self, scene_id: str) -> dict:
        folder = self.path(scene_id)
        meta = {"name": scene_id, "source": "cli", "created_at": None}
        try:
            meta.update(json.loads((folder / META_FILE).read_text()))
        except (OSError, ValueError):
            pass
        if not meta.get("created_at"):
            try:
                stamp = (folder / "scene.json").stat().st_mtime
                meta["created_at"] = datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="seconds")
            except OSError:
                pass
        return meta

    def revision(self, scene_id: str) -> int:
        """Counts the edits made since the server started. The front end reloads
        the geometry when it changes."""
        return self._revisions.get(scene_id, 0)

    def summaries(self) -> list[dict]:
        """Short description of every scene on disk, newest first."""
        out = []
        for folder in self.root.iterdir():
            if folder.name.startswith(".") or not (folder / "scene.json").is_file():
                continue
            if not _ID_RE.match(folder.name):
                continue
            try:
                doc = json.loads((folder / "scene.json").read_text())
            except (OSError, ValueError):
                continue
            meta = self.meta(folder.name)
            out.append({
                "id": folder.name,
                "name": meta.get("name") or folder.name,
                "source": meta.get("source", "cli"),
                "created_at": meta.get("created_at"),
                "n_objects": len(doc.get("objects", [])),
                "n_views": len(doc.get("cameras", [])),
                "calibrated": bool(doc.get("scale", {}).get("calibrated", False)),
                "labels": sorted({o.get("label", "object") for o in doc.get("objects", [])}),
            })
        out.sort(key=lambda s: s["created_at"] or "", reverse=True)
        return out

    # ---- access ----
    def _entry(self, scene_id: str) -> SceneEntry:
        with self._lock:
            entry = self._cache.get(scene_id)
            if entry is not None:
                self._cache.move_to_end(scene_id)
                return entry
            if not self.exists(scene_id):
                raise SceneNotFound(f"no scene '{scene_id}'")
            entry = SceneEntry(scene_id, Scene.load(self.path(scene_id)))
            self._remember(entry)
            return entry

    def _remember(self, entry: SceneEntry) -> None:
        self._cache[entry.id] = entry
        self._cache.move_to_end(entry.id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @contextmanager
    def use(self, scene_id: str) -> Iterator[SceneEntry]:
        """Exclusive access to one scene, loaded from disk when needed."""
        entry = self._entry(scene_id)
        with entry.lock:
            yield entry

    def commit(self, entry: SceneEntry) -> int:
        """Write an edited scene back to its folder. Call while holding the lock."""
        entry.scene.save(self.path(entry.id))
        with self._lock:
            self._revisions[entry.id] = self._revisions.get(entry.id, 0) + 1
            return self._revisions[entry.id]

    def put(self, scene_id: str, scene: Scene, meta: Optional[dict] = None) -> None:
        """Store a new scene under an id reserved with new_id()."""
        folder = self.path(scene_id)
        scene.save(folder)
        doc = {"name": scene_id, "source": "studio", "created_at": now_iso(), **(meta or {})}
        (folder / META_FILE).write_text(json.dumps(doc, indent=2))
        with self._lock:
            self._revisions[scene_id] = 0
            self._remember(SceneEntry(scene_id, scene))

    def discard(self, scene_id: str) -> None:
        """Remove a reserved folder that never received a scene (a failed job)."""
        folder = self.path(scene_id)
        if folder.is_dir() and not (folder / "scene.json").exists():
            shutil.rmtree(folder, ignore_errors=True)

    def delete(self, scene_id: str) -> None:
        if not self.exists(scene_id):
            raise SceneNotFound(f"no scene '{scene_id}'")
        with self._lock:
            self._cache.pop(scene_id, None)
            self._revisions.pop(scene_id, None)
        shutil.rmtree(self.path(scene_id))
