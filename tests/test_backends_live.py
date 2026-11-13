# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the real model backends. Skipped unless they are installed.

    SCENELIFT_TEST_INPUT=/path/to/room.mp4 pytest -q tests/test_backends_live.py
"""

import os

import pytest

mapanything = pytest.importorskip("mapanything")
sam3 = pytest.importorskip("sam3")


def test_real_pipeline(tmp_path):
    src = os.environ.get("SCENELIFT_TEST_INPUT")
    if not src:
        pytest.skip("set SCENELIFT_TEST_INPUT to a video or a folder of photos")
    from scenelift import Config, SceneLift

    cfg = Config(max_frames=8)
    scene = SceneLift.default(cfg, sam3d_repo=os.environ.get("SAM3D_REPO")).run(src)
    assert len(scene) > 0
    scene.save(tmp_path / "scene")
    scene.export(tmp_path / "scene.glb")
    for row in scene.summary():
        assert all(0.01 < s < 20 for s in row["size_m"]), row
