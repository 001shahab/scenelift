# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Everything that can be configured about a Studio server, in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_SCENES_DIR = "scenes"


@dataclass
class StudioSettings:
    """Each field can also be set by the environment variable named next to it."""

    scenes_dir: Path = field(default_factory=lambda: Path(DEFAULT_SCENES_DIR))  # SCENELIFT_SCENES_DIR
    host: str = DEFAULT_HOST  # SCENELIFT_HOST
    port: int = DEFAULT_PORT  # SCENELIFT_PORT
    device: Optional[str] = None  # SCENELIFT_DEVICE; None lets the backends choose (cuda, mps, cpu)
    sam3d_repo: Optional[str] = None  # SAM3D_REPO; a sam-3d-objects checkout enables completion
    max_upload_mb: int = 2048  # SCENELIFT_MAX_UPLOAD_MB; total size of one upload
    scene_cache: int = 6  # scenes kept in memory
    open_browser: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "StudioSettings":
        """Settings from the environment; keyword arguments that are not None win."""
        env = os.environ
        values = {
            "scenes_dir": Path(env.get("SCENELIFT_SCENES_DIR", DEFAULT_SCENES_DIR)),
            "host": env.get("SCENELIFT_HOST", DEFAULT_HOST),
            "port": int(env.get("SCENELIFT_PORT", DEFAULT_PORT)),
            "device": env.get("SCENELIFT_DEVICE") or None,
            "sam3d_repo": env.get("SAM3D_REPO") or None,
            "max_upload_mb": int(env.get("SCENELIFT_MAX_UPLOAD_MB", 2048)),
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = Path(value) if key == "scenes_dir" else value
        return cls(**values)
