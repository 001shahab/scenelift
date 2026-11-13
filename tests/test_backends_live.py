# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the real model backends. Skipped unless they are installed
and SCENELIFT_TEST_INPUT points at a video or a folder of photos.

    SCENELIFT_TEST_INPUT=/path/to/room.mp4 pytest -q tests/test_backends_live.py

The two geometry tests need MapAnything only. The full pipeline test also
needs SAM 3.
"""

import copy
import os

import numpy as np
import pytest

pytest.importorskip("mapanything")

SRC = os.environ.get("SCENELIFT_TEST_INPUT")
pytestmark = pytest.mark.skipif(not SRC, reason="set SCENELIFT_TEST_INPUT to a video or a folder of photos")


@pytest.fixture(scope="module")
def footage():
    """Keyframes of the input and what MapAnything makes of them."""
    from scenelift import Config, ingest
    from scenelift.backends.mapanything import MapAnythingBackend

    frames = ingest.load(SRC, Config(max_frames=8))
    return frames, MapAnythingBackend().infer(frames)


def test_geometry_backend_contract(footage):
    """Every View must be what the rest of the pipeline assumes (docs/BACKENDS.md)."""
    frames, views = footage
    assert len(views) == len(frames) > 0
    for frame, v in zip(frames, views):
        H, W = v.size
        assert v.index == frame.index
        assert v.image.shape == (H, W, 3) and v.image.dtype == np.uint8
        assert v.depth.shape == (H, W) and v.depth.dtype == np.float32
        assert v.conf.shape == (H, W) and v.valid.shape == (H, W) and v.valid.dtype == bool
        assert 0.0 <= v.conf.min() and v.conf.max() <= 1.0
        assert v.valid.mean() > 0.3, "most of an indoor frame should have usable depth"
        depth = v.depth[v.valid]
        assert np.isfinite(depth).all() and depth.min() > 0
        assert 0.05 < np.median(depth) < 50, "depth should be in metres"
        assert v.K.shape == (3, 3) and v.K[0, 0] > 0 and v.K[1, 1] > 0
        assert 0 < v.K[0, 2] < W and 0 < v.K[1, 2] < H
        assert v.cam2world.shape == (4, 4)
        R = v.cam2world[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-3), "cam2world must hold a rotation"
        assert np.allclose(v.cam2world[3], [0, 0, 0, 1], atol=1e-6)


def test_geometry_through_the_core(footage, tmp_path):
    """Real depth through scale harmonisation, gravity, layout, fusion and
    saving, with no segmenter: the scene is background only."""
    from scenelift import Config, Scene, SceneLift

    frames, views = footage

    class Cached:
        def infer(self, _frames):
            return copy.deepcopy(views)

    class NoMasks:
        def segment(self, vs):
            return [[] for _ in vs]

    scene = SceneLift(Cached(), NoMasks(), None, Config(max_frames=8)).run_frames(frames)
    assert len(scene) == 0
    assert len(scene.background_points) > 1000
    assert scene.scale.n_views == len(views)
    assert 0 <= scene.scale.sigma_consistency < 0.5
    scene.save(tmp_path / "scene")
    again = Scene.load(tmp_path / "scene")
    assert len(again.cameras) == len(views)
    assert len(again.background_points) == len(scene.background_points)


def test_real_pipeline(tmp_path):
    pytest.importorskip("sam3")
    from scenelift import Config, SceneLift

    cfg = Config(max_frames=8)
    scene = SceneLift.default(cfg, sam3d_repo=os.environ.get("SAM3D_REPO")).run(SRC)
    assert len(scene) > 0
    scene.save(tmp_path / "scene")
    scene.export(tmp_path / "scene.glb")
    for row in scene.summary():
        assert all(0.01 < s < 20 for s in row["size_m"]), row
