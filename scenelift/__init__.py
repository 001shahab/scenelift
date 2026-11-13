# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""scenelift: casual phone video to a metric, editable, object-level 3D scene.

    from scenelift import SceneLift, query, select

    scene = SceneLift.default().run("room.mp4")
    query.distance(scene, "sofa_1", "table_1")
    scene.remove("table_1")
    scene.export("room.glb")

The method is described step by step in docs/DESIGN.md.
"""

from .config import Config
from .pipeline import SceneLift
from .scene import Scene, SceneObject
from .types import Frame, Mask2D, View

__version__ = "0.1.0"
__author__ = "Prof. Shahab Anbarjafari"
__email__ = "shb@3sholding.com"
__copyright__ = "Copyright 2026 3S Holding OÜ"
__license__ = "Apache-2.0"

__all__ = ["Config", "SceneLift", "Scene", "SceneObject", "Frame", "Mask2D", "View", "__version__"]
