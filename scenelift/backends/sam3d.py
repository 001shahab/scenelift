# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Completion backend built on SAM 3D Objects (Meta, "SAM License").

Install:  follow https://github.com/facebookresearch/sam-3d-objects (needs a CUDA GPU)
          and download the checkpoints into <repo>/checkpoints/hf.
Licence:  SAM License, not an OSI licence. Not bundled with scenelift.

Only the generated mesh is used. Its pose and scale are discarded on purpose:
scenelift registers the mesh to its own multi-view metric points, which is
more reliable than a single-image layout guess. The call sequence follows
demo.py of the SAM 3D Objects repo (checked November 2026); this wrapper has
not been run on real weights by its authors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import CompletedShape, Mask2D, View


class Sam3DObjectsBackend:
    def __init__(self, repo_dir: str, tag: str = "hf", seed: int = 42):
        repo = Path(repo_dir)
        sys.path.insert(0, str(repo / "notebook"))
        from inference import Inference  # provided by the SAM 3D Objects repo

        self.seed = seed
        self.inference = Inference(str(repo / "checkpoints" / tag / "pipeline.yaml"), compile=False)

    def complete(self, view: View, mask: Mask2D) -> Optional[CompletedShape]:
        import trimesh

        out = self.inference(view.image, mask.mask, seed=self.seed)
        mesh = out.get("glb")
        if mesh is None:
            return None
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)
        colors = None
        try:
            vc = np.asarray(mesh.visual.vertex_colors)
            if vc.ndim == 2 and len(vc) == len(mesh.vertices):
                colors = vc[:, :3].astype(np.uint8)
        except Exception:
            pass
        return CompletedShape(vertices=np.asarray(mesh.vertices, dtype=np.float64),
                              faces=np.asarray(mesh.faces, dtype=np.int64), colors=colors)
