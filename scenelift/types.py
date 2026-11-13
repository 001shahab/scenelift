# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Plain data containers shared by every stage of the pipeline.

Conventions used everywhere in scenelift:
  * Units are metres.
  * Camera frame is OpenCV: +X right, +Y down, +Z forward.
  * ``cam2world`` is a 4x4 matrix mapping camera coordinates to world.
  * After gravity alignment the world is Z-up with the floor at z = 0.
  * Pixel (u, v) = (column, row); integer coordinates are pixel centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """One input image, before any model has seen it."""

    index: int
    image: np.ndarray  # (H, W, 3) uint8 RGB
    source: Optional[str] = None  # file path or "video.mp4#123"
    timestamp: Optional[float] = None  # seconds, for video


@dataclass
class View:
    """One frame after the geometry backend: image + metric depth + camera."""

    index: int
    image: np.ndarray  # (H, W, 3) uint8 RGB, at depth resolution
    depth: np.ndarray  # (H, W) float32, z-depth in metres
    K: np.ndarray  # (3, 3) pinhole intrinsics at depth resolution
    cam2world: np.ndarray  # (4, 4)
    conf: np.ndarray  # (H, W) float32 in [0, 1], higher is better
    valid: np.ndarray  # (H, W) bool
    source: Optional[str] = None

    @property
    def world2cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam2world)

    @property
    def size(self) -> tuple[int, int]:
        return self.depth.shape[0], self.depth.shape[1]

    @property
    def center(self) -> np.ndarray:
        return self.cam2world[:3, 3]


@dataclass
class Mask2D:
    """One 2D instance mask in one view."""

    view_index: int
    mask: np.ndarray  # (H, W) bool, same resolution as the view
    label: str = "object"
    score: float = 1.0
    track_id: Optional[int] = None  # set when the segmenter tracks objects across frames


@dataclass
class CompletedShape:
    """A complete mesh proposed by a generative backend for one object.

    The mesh lives in the backend's own canonical frame with arbitrary scale.
    scenelift never trusts that frame: registration to the observed points
    decides the final pose and scale (see completion.py).
    """

    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray  # (F, 3) int
    colors: Optional[np.ndarray] = None  # (V, 3) uint8
    pose_cam: Optional[np.ndarray] = None  # optional 4x4 similarity, mesh -> camera


@dataclass
class Instance3D:
    """A multi-view object track: the same physical object seen in several views."""

    id: int
    observations: list[Mask2D] = field(default_factory=list)
    voxels: Optional[np.ndarray] = None  # sorted unique packed voxel keys
    label_votes: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)  # per observation: (points, colors, conf)
    track_id: Optional[int] = None

    @property
    def label(self) -> str:
        if not self.label_votes:
            return "object"
        return max(self.label_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def n_views(self) -> int:
        return len({o.view_index for o in self.observations})
