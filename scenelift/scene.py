# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""The scene representation: what scenelift produces and what users edit.

A Scene is a set of separate objects, a simple room layout (floor and walls),
a background point cloud, the cameras, and the scale information. Every object
keeps two layers of geometry:

  * ``points``: the surface that was actually measured by the camera.
  * ``mesh_*``: an optional complete mesh, where every vertex is tagged as
    observed (close to a measured point) or generated (filled in by a model).

Keeping the two apart is what lets queries say "this number relies on a part
of the object nobody saw".
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image

from .align import Layout
from .geometry import gravity_obb, make_transform, rot_z, transform_points
from .scale import ScaleInfo

FORMAT_VERSION = 1


@dataclass
class SceneObject:
    id: str
    label: str
    points: np.ndarray  # (N, 3) measured surface, world frame
    colors: np.ndarray  # (N, 3) uint8
    mesh_vertices: Optional[np.ndarray] = None  # (V, 3)
    mesh_faces: Optional[np.ndarray] = None  # (F, 3)
    mesh_colors: Optional[np.ndarray] = None  # (V, 3) uint8
    vertex_observed: Optional[np.ndarray] = None  # (V,) bool
    completion: str = "none"  # "none" | "hull" | "generated"
    support: Optional[str] = None  # "floor", "wall", another object's id, or None
    flags: list = field(default_factory=list)
    n_views: int = 1
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # edits applied so far
    obb: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)  # quality of the completion fit, for diagnostics

    def __post_init__(self):
        if not self.obb:
            self.update_obb()

    # ---- geometry ----
    @property
    def has_mesh(self) -> bool:
        return self.mesh_vertices is not None and self.mesh_faces is not None

    def extent_points(self) -> np.ndarray:
        """Points that define the full size of the object."""
        return self.mesh_vertices if self.has_mesh else self.points

    def update_obb(self) -> None:
        self.obb = gravity_obb(self.extent_points())

    @property
    def bottom(self) -> float:
        return float(self.obb["center"][2] - self.obb["size"][2] / 2)

    @property
    def top(self) -> float:
        return float(self.obb["center"][2] + self.obb["size"][2] / 2)

    @property
    def observed_fraction(self) -> float:
        """Share of the final surface that was measured rather than generated."""
        if self.vertex_observed is None or len(self.vertex_observed) == 0:
            return 1.0
        return float(np.mean(self.vertex_observed))

    def apply(self, T: np.ndarray) -> None:
        """Move the object by a rigid or similarity transform."""
        self.points = transform_points(T, self.points)
        if self.has_mesh:
            self.mesh_vertices = transform_points(T, self.mesh_vertices)
        self.transform = T @ self.transform
        self.update_obb()

    def surface_samples(self, n: int = 4000, rng: Optional[np.random.Generator] = None):
        """Sample the surface. Returns (points (n,3), observed (n,) bool)."""
        rng = rng or np.random.default_rng(0)
        if self.has_mesh and len(self.mesh_faces) > 0:
            v, f = self.mesh_vertices, self.mesh_faces
            tri = v[f]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            if area.sum() > 0:
                fi = rng.choice(len(f), size=n, p=area / area.sum())
                r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
                w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
                pts = (tri[fi] * w[:, :, None]).sum(axis=1)
                if self.vertex_observed is not None:
                    obs = self.vertex_observed[f[fi]].sum(axis=1) >= 2
                else:
                    obs = np.ones(n, dtype=bool)
                return pts, obs
        pts = self.points
        if len(pts) > n:
            pts = pts[rng.choice(len(pts), n, replace=False)]
        return pts, np.ones(len(pts), dtype=bool)

    def to_trimesh(self) -> trimesh.Trimesh:
        if self.has_mesh:
            return trimesh.Trimesh(self.mesh_vertices, self.mesh_faces,
                                   vertex_colors=self.mesh_colors, process=False)
        return trimesh.Trimesh(self.points, process=False).convex_hull

    def meta(self) -> dict:
        return {
            "id": self.id, "label": self.label, "completion": self.completion,
            "support": self.support, "flags": list(self.flags), "n_views": int(self.n_views),
            "observed_fraction": round(self.observed_fraction, 4),
            "fit": dict(self.fit), "transform": self.transform.tolist(),
            "obb": {"center": self.obb["center"].tolist(), "size": self.obb["size"].tolist(),
                    "yaw": float(self.obb["yaw"])},
        }


class Scene:
    def __init__(self, objects: Optional[dict] = None, layout: Optional[Layout] = None,
                 background_points: Optional[np.ndarray] = None,
                 background_colors: Optional[np.ndarray] = None,
                 scale: Optional[ScaleInfo] = None, cameras: Optional[list] = None,
                 config: Optional[dict] = None):
        self.objects: dict[str, SceneObject] = objects or {}
        self.layout = layout or Layout()
        self.background_points = background_points if background_points is not None else np.zeros((0, 3))
        self.background_colors = (background_colors if background_colors is not None
                                  else np.zeros((0, 3), np.uint8))
        self.scale = scale or ScaleInfo()
        # each camera: dict(K, cam2world, size (H, W), source, depth (optional),
        # image (optional, the keyframe in memory), frame (optional, path of the stored keyframe))
        self.cameras: list = cameras or []
        self.config = config or {}
        self.history: list = []

    # ---- lookup ----
    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, obj_id: str) -> SceneObject:
        if obj_id not in self.objects:
            raise KeyError(f"no object '{obj_id}'. Available: {', '.join(self.objects) or 'none'}")
        return self.objects[obj_id]

    @property
    def ids(self) -> list:
        return list(self.objects)

    def find(self, text: str) -> list:
        """Ids whose id or label matches the text. Exact matches come first."""
        t = text.lower().strip()
        exact = [o.id for o in self.objects.values() if t in (o.id.lower(), o.label.lower())]
        if exact:
            return exact
        words = set(t.replace("_", " ").split())
        return [o.id for o in self.objects.values()
                if t in o.label.lower() or o.label.lower() in t
                or words & set(o.label.lower().split())]

    def dependants(self, obj_id: str) -> list:
        """Everything resting on this object, directly or through other objects."""
        out, stack = [], [obj_id]
        while stack:
            cur = stack.pop()
            for o in self.objects.values():
                if o.support == cur and o.id not in out:
                    out.append(o.id)
                    stack.append(o.id)
        return out

    def summary(self) -> list:
        rows = []
        for o in self.objects.values():
            L, W, H = o.obb["size"]
            # plain Python numbers, so a row prints and serialises cleanly
            rows.append({"id": o.id, "label": o.label,
                         "size_m": [round(float(L), 3), round(float(W), 3), round(float(H), 3)],
                         "position_m": [round(float(c), 3) for c in o.obb["center"]],
                         "support": o.support, "views": int(o.n_views),
                         "observed_fraction": round(float(o.observed_fraction), 2), "flags": list(o.flags)})
        return rows

    # ---- editing ----
    def _log(self, op: str, **kw) -> None:
        self.history.append({"op": op, **kw})

    def settle(self) -> None:
        from .physics import settle
        settle(self)

    def remove(self, obj_id: str, settle: bool = True) -> SceneObject:
        """Delete an object. With settle=True the things that stood on it drop
        onto whatever is underneath."""
        obj = self[obj_id]
        del self.objects[obj_id]
        self._log("remove", id=obj_id)
        if settle:
            self.settle()
        return obj

    def translate(self, obj_id: str, delta, carry: bool = True) -> None:
        """Move an object by delta = (dx, dy, dz) metres. carry=True also moves
        whatever rests on it."""
        T = make_transform(t=np.asarray(delta, dtype=float))
        for i in [obj_id] + (self.dependants(obj_id) if carry else []):
            self[i].apply(T)
        self._log("translate", id=obj_id, delta=[float(x) for x in delta], carry=carry)

    def rotate(self, obj_id: str, yaw_deg: float, carry: bool = True) -> None:
        """Turn an object about the vertical axis through its centre."""
        c = self[obj_id].obb["center"]
        R = rot_z(np.deg2rad(yaw_deg))
        T = make_transform(R, c - R @ c)
        for i in [obj_id] + (self.dependants(obj_id) if carry else []):
            self[i].apply(T)
        self._log("rotate", id=obj_id, yaw_deg=float(yaw_deg), carry=carry)

    def duplicate(self, obj_id: str, offset=(0.5, 0.0, 0.0)) -> str:
        src = self[obj_id]
        n = 2
        while f"{src.label}_{n}".replace(" ", "_") in self.objects:
            n += 1
        new = copy.deepcopy(src)
        new.id = f"{src.label}_{n}".replace(" ", "_")
        new.apply(make_transform(t=np.asarray(offset, dtype=float)))
        self.objects[new.id] = new
        self._log("duplicate", id=obj_id, new_id=new.id)
        return new.id

    def calibrate(self, measured: float, true: float, sigma_true: float = 0.005,
                  sigma_measured: float = 0.012) -> float:
        """Fix the absolute scale with one known length.

        measured: the length as the scene currently reports it (metres).
        true:     the real length, for example from a tape measure.
        sigma_true: how well the real length is known (metres).
        sigma_measured: how well the scene resolves that length (metres).
        Rescales everything about the origin and returns the factor applied.
        """
        f = float(true) / float(measured)
        S = make_transform(s=f)
        for o in self.objects.values():
            o.apply(S)
        self.background_points = self.background_points * f
        self.layout.floor_quad = self.layout.floor_quad * f
        for w in self.layout.walls:
            w["quad"] = w["quad"] * f
            w["d"] = w["d"] * f
        for cam in self.cameras:
            cam["cam2world"] = np.array(cam["cam2world"], dtype=float)
            cam["cam2world"][:3, 3] *= f
            if cam.get("depth") is not None:
                cam["depth"] = (cam["depth"].astype(np.float32) * f).astype(np.float32)
        self.scale.calibrated = True
        self.scale.sigma_reference = float(np.hypot(sigma_true, sigma_measured)) / float(true)
        self.scale.global_factor *= f
        self._log("calibrate", factor=f)
        return f

    # ---- keyframes ----
    def frame_image(self, camera_index: int) -> Optional[np.ndarray]:
        """The keyframe seen by one camera as (H, W, 3) uint8 RGB, or None when
        the scene was saved without frames."""
        cam = self.cameras[camera_index]
        if cam.get("image") is not None:
            return np.asarray(cam["image"])
        if cam.get("frame") and Path(cam["frame"]).is_file():
            return np.asarray(Image.open(cam["frame"]).convert("RGB"))
        return None

    @staticmethod
    def _save_frame(root: Path, index: int, cam: dict) -> Optional[str]:
        """Store one camera's keyframe as frames/<index>.jpg. Returns the path
        relative to the scene folder, or None when the camera has no keyframe.

        A keyframe that is already on disk is copied, not re-encoded, so saving
        a scene many times never degrades it.
        """
        rel = f"frames/{index:04d}.jpg"
        dst = root / rel
        stored, image = cam.get("frame"), cam.get("image")
        if stored and Path(stored).is_file():
            if Path(stored).resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stored, dst)
        elif image is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(dst, quality=92)
        else:
            return None
        cam["frame"] = str(dst)
        return rel

    # ---- persistence ----
    def save(self, path) -> None:
        """Write a scene folder: scene.json plus .npz arrays and the keyframes."""
        root = Path(path)
        (root / "objects").mkdir(parents=True, exist_ok=True)
        for o in self.objects.values():
            arrays = {"points": o.points.astype(np.float32), "colors": o.colors.astype(np.uint8)}
            if o.has_mesh:
                arrays["mesh_vertices"] = o.mesh_vertices.astype(np.float32)
                arrays["mesh_faces"] = o.mesh_faces.astype(np.int32)
                if o.mesh_colors is not None:
                    arrays["mesh_colors"] = o.mesh_colors.astype(np.uint8)
                if o.vertex_observed is not None:
                    arrays["vertex_observed"] = o.vertex_observed.astype(bool)
            np.savez_compressed(root / "objects" / f"{o.id}.npz", **arrays)
        np.savez_compressed(root / "background.npz", points=self.background_points.astype(np.float32),
                            colors=self.background_colors.astype(np.uint8))
        cams = []
        depth = {}
        for i, c in enumerate(self.cameras):
            cams.append({"K": np.asarray(c["K"]).tolist(),
                         "cam2world": np.asarray(c["cam2world"]).tolist(),
                         "size": [int(c["size"][0]), int(c["size"][1])],
                         "source": c.get("source"),
                         "frame": self._save_frame(root, i, c)})
            if c.get("depth") is not None:
                depth[f"d{i}"] = c["depth"].astype(np.float16)
        if depth:
            np.savez_compressed(root / "depth.npz", **depth)
        doc = {"format_version": FORMAT_VERSION, "units": "metres", "up_axis": "+Z",
               "objects": [o.meta() for o in self.objects.values()],
               "layout": self.layout.to_dict(), "scale": self.scale.to_dict(),
               "cameras": cams, "config": self.config, "history": self.history}
        (root / "scene.json").write_text(json.dumps(doc, indent=2))

    @classmethod
    def load(cls, path) -> "Scene":
        root = Path(path)
        doc = json.loads((root / "scene.json").read_text())
        objects = {}
        for m in doc["objects"]:
            a = np.load(root / "objects" / f"{m['id']}.npz")
            o = SceneObject(
                id=m["id"], label=m["label"], points=a["points"].astype(np.float64), colors=a["colors"],
                mesh_vertices=a["mesh_vertices"].astype(np.float64) if "mesh_vertices" in a else None,
                mesh_faces=a["mesh_faces"] if "mesh_faces" in a else None,
                mesh_colors=a["mesh_colors"] if "mesh_colors" in a else None,
                vertex_observed=a["vertex_observed"] if "vertex_observed" in a else None,
                completion=m["completion"], support=m["support"], flags=m["flags"],
                n_views=m["n_views"], transform=np.array(m["transform"]), fit=m.get("fit", {}))
            objects[o.id] = o
        bg = np.load(root / "background.npz")
        cams = [{"K": np.array(c["K"]), "cam2world": np.array(c["cam2world"]),
                 "size": tuple(c["size"]), "source": c.get("source"), "depth": None, "image": None,
                 "frame": str(root / c["frame"]) if c.get("frame") and (root / c["frame"]).is_file() else None}
                for c in doc["cameras"]]
        if (root / "depth.npz").exists():
            d = np.load(root / "depth.npz")
            for i, c in enumerate(cams):
                if f"d{i}" in d:
                    c["depth"] = d[f"d{i}"].astype(np.float32)
        scene = cls(objects, Layout.from_dict(doc["layout"]), bg["points"].astype(np.float64),
                    bg["colors"], ScaleInfo.from_dict(doc["scale"]), cams, doc.get("config", {}))
        scene.history = doc.get("history", [])
        return scene

    # ---- export ----
    def to_trimesh_scene(self, include_layout: bool = True, include_background: bool = False) -> trimesh.Scene:
        """One named node per object, so Blender, Unity and others import them separately."""
        ts = trimesh.Scene()
        for o in self.objects.values():
            ts.add_geometry(o.to_trimesh(), node_name=o.id, geom_name=o.id)
        if include_layout:
            quads = [("floor", self.layout.floor_quad)] + [
                (f"wall_{i}", w["quad"]) for i, w in enumerate(self.layout.walls)]
            for name, q in quads:
                if np.abs(q).sum() == 0:
                    continue
                m = trimesh.Trimesh(q, [[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], process=False)
                m.visual.vertex_colors = np.tile([200, 200, 200, 255], (4, 1))
                ts.add_geometry(m, node_name=name, geom_name=name)
        if include_background and len(self.background_points):
            ts.add_geometry(trimesh.PointCloud(self.background_points, self.background_colors),
                            node_name="background", geom_name="background")
        return ts

    def export(self, path, include_layout: bool = True, include_background: bool = False) -> None:
        """Export by file extension: .glb / .gltf (Y-up, as the format requires),
        .obj, .ply (Z-up, metres)."""
        path = str(path)
        ts = self.to_trimesh_scene(include_layout, include_background)
        if path.lower().endswith((".glb", ".gltf")):
            z_up_to_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
            ts.apply_transform(z_up_to_y_up)
        ts.export(path)

    def export_points(self, path) -> None:
        """All measured points (objects + background) as one coloured PLY."""
        pts = [o.points for o in self.objects.values()] + [self.background_points]
        col = [o.colors for o in self.objects.values()] + [self.background_colors]
        trimesh.PointCloud(np.concatenate(pts), np.concatenate(col)).export(str(path))
