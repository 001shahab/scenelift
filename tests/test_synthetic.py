# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test on a synthetic room with exact ground truth.

Runs in a few seconds on a laptop CPU, needs no model weights:
    pytest -q
"""

import json

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from scenelift import Config, Scene, SceneLift, query, select
from scenelift.backends.synthetic import (SyntheticCompleter, SyntheticGeometry, SyntheticSegmenter,
                                          SyntheticWorld)
from scenelift.cli import main as cli
from scenelift.eval import evaluate
from scenelift.geometry import rot_z
from scenelift.types import CompletedShape


@pytest.fixture(scope="module")
def world():
    return SyntheticWorld(seed=0)


@pytest.fixture(scope="module")
def scene(world):
    lift = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), SyntheticCompleter(world), Config())
    return lift.run_frames(world.frames())


def fresh(scene, tmp_path):
    scene.save(tmp_path / "s")
    return Scene.load(tmp_path / "s")


def true_gap(world, a, b, n=40000):
    rng = np.random.default_rng(1)
    clouds = []
    for name in (a, b):
        bx = next(x for x in world.boxes if x.name == name)
        m = trimesh.creation.box(extents=bx.size)
        p = np.asarray(trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 30)))[0])
        clouds.append(p @ rot_z(bx.yaw).T + bx.center)
    return float(cKDTree(clouds[1]).query(clouds[0])[0].min())


def test_objects_found(scene):
    assert sorted(scene.ids) == ["box_1", "cabinet_1", "table_1"]
    assert scene.layout.floor_found
    assert len(scene.layout.walls) == 2


def test_scale_harmonisation(world, scene):
    # the corrections must undo the injected per-view errors (up to one shared factor)
    resid = np.log(np.array(scene.scale.per_view) * world.scale_factors)
    assert np.std(resid) < 0.006
    assert 0.02 < scene.scale.sigma_consistency < 0.09


def test_gravity_alignment(scene):
    assert abs(scene["table_1"].bottom) < 0.01
    assert abs(scene["cabinet_1"].bottom) < 0.01
    for w in scene.layout.walls:  # walls are vertical and square to the axes
        assert abs(w["normal"][2]) < 1e-6
        assert max(abs(w["normal"][0]), abs(w["normal"][1])) > 0.999


@pytest.mark.parametrize("name", ["table", "box", "cabinet"])
def test_metric_dimensions(world, scene, name):
    got = scene[f"{name}_1"].obb["size"]
    true = world.true_dims(name)
    assert np.all(np.abs(got - true) / true < 0.05), (got, true)
    assert scene[f"{name}_1"].completion == "generated"


def test_dimension_intervals_cover_truth(world, scene):
    for name in ("table", "box", "cabinet"):
        dims = query.dimensions(scene, f"{name}_1")
        for m, t in zip(dims.values(), world.true_dims(name)):
            lo, hi = m.interval95
            assert lo <= t <= hi, (name, str(m), t)


def test_distance(world, scene):
    m = query.distance(scene, "table_1", "cabinet_1")
    t = true_gap(world, "table", "cabinet")
    assert abs(m.value - t) < 0.05, (str(m), t)
    assert m.interval95[0] <= t <= m.interval95[1]


def test_provenance(scene):
    cab = scene["cabinet_1"]
    assert 0.2 < cab.observed_fraction < 0.9  # the back and the bottom were never seen
    assert cab.vertex_observed.dtype == bool


def test_support_graph(scene):
    assert scene["table_1"].support == "floor"
    assert scene["cabinet_1"].support == "floor"
    assert scene["box_1"].support == "table_1"
    assert abs(scene["box_1"].bottom - scene["table_1"].top) < 0.005


def test_edit_carry_and_drop(scene, tmp_path):
    s = fresh(scene, tmp_path)
    before = s["box_1"].obb["center"].copy()
    s.translate("table_1", (0.3, 0.0, 0.0))
    assert np.allclose(s["box_1"].obb["center"] - before, [0.3, 0, 0], atol=1e-3)  # rides along
    s.remove("table_1")
    assert "table_1" not in s.objects
    assert s["box_1"].support == "floor" and abs(s["box_1"].bottom) < 0.005  # falls to the floor
    new = s.duplicate("cabinet_1", (0.0, -1.0, 0.0))
    assert new in s.objects and len(s) == 3


def test_fit_queries(world, scene):
    L, W, H = world.true_dims("cabinet")
    assert query.fits_in_space(scene, "cabinet_1", (L + 0.25, W + 0.25, H + 0.25)).verdict == "yes"
    assert query.fits_in_space(scene, "cabinet_1", (L - 0.15, W - 0.15, H)).verdict == "no"
    assert query.fits_in_space(scene, "cabinet_1", (L + 0.005, W + 0.005, H + 0.005)).verdict == "uncertain"
    assert query.fits_between(scene, "box_1", "table_1", "cabinet_1").verdict == "yes"


def test_calibration(world, scene, tmp_path):
    s = fresh(scene, tmp_path)
    measured = query.dimensions(s, "cabinet_1")["height"].value
    s.calibrate(measured, 1.0)
    assert abs(s["cabinet_1"].obb["size"][2] - 1.0) < 1e-6
    assert s.scale.calibrated and s.scale.sigma_rel < scene.scale.sigma_rel


def test_placements(scene):
    res = query.find_placements(scene, "box_1")
    assert res and res[0]["spare_m"] > 0.05
    xy = np.array(res[0]["center_xy"])
    for o in scene.objects.values():  # the proposed spot is not inside another object
        if o.id != "box_1" and o.support == "floor":
            assert np.linalg.norm(xy - o.obb["center"][:2]) > 0.2


def test_click_box_text(world, scene):
    ids = world.render_all()[4][1]
    for k, b in enumerate(world.boxes):
        v, u = np.argwhere(ids == k).mean(axis=0)
        if ids[int(v), int(u)] == k:
            assert select.by_click(scene, 4, u, v) == f"{b.name}_1"
    assert "table_1" in select.by_box(scene, 4, (0, 0, world.W - 1, world.H - 1))
    assert select.by_text(scene, "cabinet") == ["cabinet_1"]


def test_save_load_export(scene, tmp_path):
    s = fresh(scene, tmp_path)
    assert s.ids == scene.ids
    assert np.allclose(s["table_1"].obb["size"], scene["table_1"].obb["size"], atol=1e-4)
    s.export(tmp_path / "scene.glb")
    loaded = trimesh.load(tmp_path / "scene.glb")
    assert {"table_1", "box_1", "cabinet_1", "floor"} <= set(loaded.geometry.keys())
    s.export_points(tmp_path / "points.ply")
    assert (tmp_path / "points.ply").stat().st_size > 1000


def test_tools_and_eval(world, scene):
    out = query.call_tool(scene, "measure_distance", {"a": "table", "b": "cabinet"})
    json.dumps(out)
    assert out["value_m"] > 0
    gt = [{"type": "dimension", "object": f"{n}_1", "axis": ax, "true_m": float(t)}
          for n in ("table", "box", "cabinet") for ax, t in zip(("length", "width", "height"), world.true_dims(n))]
    gt.append({"type": "fit", "object": "box_1", "space": [1, 1, 1], "true": True})
    res = evaluate(scene, gt)
    assert res["abs_rel_error"] < 0.04 and res["coverage_95"] == 1.0 and res["fit_accuracy"] == 1.0


def test_cli(scene, tmp_path, capsys):
    scene.save(tmp_path / "s")
    p = str(tmp_path / "s")
    assert cli(["info", p]) == 0
    assert cli(["measure", p, "table_1", "cabinet_1"]) == 0
    assert cli(["size", p, "box_1"]) == 0
    assert cli(["fit", p, "box_1", "--space", "1", "1", "1"]) == 0
    assert cli(["place", p, "box_1"]) == 0
    assert cli(["edit", p, "rotate", "cabinet_1", "90"]) == 0
    assert cli(["export", p, str(tmp_path / "o.glb")]) == 0
    assert "table_1" in capsys.readouterr().out


def test_wrong_shape_is_rejected(world):
    class Sphere:
        def complete(self, view, mask):
            m = trimesh.creation.icosphere(subdivisions=3)
            return CompletedShape(np.asarray(m.vertices), np.asarray(m.faces))

    s = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), Sphere(), Config()).run_frames(world.frames())
    for o in s.objects.values():  # a sphere is not a box: fall back to measured geometry and say so
        assert o.completion == "hull" and "completion_rejected" in o.flags


@pytest.mark.parametrize("n_views", [2, 3, 6])
def test_few_views(n_views):
    w = SyntheticWorld(n_views=n_views, seed=3, depth_noise=0.006)
    s = SceneLift(SyntheticGeometry(w), SyntheticSegmenter(w), SyntheticCompleter(w), Config()).run_frames(w.frames())
    assert sorted(s.ids) == ["box_1", "cabinet_1", "table_1"]  # no duplicates across wide baselines
    assert s["box_1"].support == "table_1"
    err = np.abs(s["table_1"].obb["size"] / w.true_dims("table") - 1)
    # two photos 130 degrees apart share little surface, so the bound is looser there
    assert err.max() < (0.12 if n_views == 2 else 0.08), err


def test_without_completer_uses_hull(world):
    s = SceneLift(SyntheticGeometry(world), SyntheticSegmenter(world), None, Config()).run_frames(world.frames())
    assert all(o.completion == "hull" for o in s.objects.values())
    assert s["box_1"].support == "table_1"
