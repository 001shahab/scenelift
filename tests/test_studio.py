# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Tests of the Studio REST API on the synthetic room.

They need no model weights and no browser:
    pytest -q tests/test_studio.py
"""

import base64
import io
import time
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from scenelift import Scene, query  # noqa: E402
from scenelift.studio import StudioSettings, create_app  # noqa: E402
from scenelift.studio.store import SceneStore, slugify  # noqa: E402

IDS = ["box_1", "cabinet_1", "table_1"]


def wait_for(client, job: dict, timeout: float = 180.0) -> dict:
    """Poll a job until it leaves the queue. The pipeline runs in a worker thread."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job did not finish in {timeout} s: {job}")


def build_demo(client, **overrides) -> str:
    res = client.post("/api/scenes/demo", json={"name": "Test room", **overrides})
    assert res.status_code == 202, res.text
    job = wait_for(client, res.json())
    assert job["status"] == "done", job["error"]
    assert all(s["status"] == "done" for s in job["stages"])
    return job["scene_id"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = StudioSettings(scenes_dir=tmp_path_factory.mktemp("scenes"), open_browser=False)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def demo(client) -> str:
    """One synthetic room shared by the read-only tests."""
    return build_demo(client)


@pytest.fixture()
def scratch(client):
    """A room of its own for a test that edits the scene."""
    scene_id = build_demo(client, name="Scratch")
    yield scene_id
    client.delete(f"/api/scenes/{scene_id}")


def objects_of(doc: dict) -> dict:
    return {o["id"]: o for o in doc["objects"]}


# --------------------------------------------------------------------------
# system and library
# --------------------------------------------------------------------------
def test_health_and_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    system = client.get("/api/system").json()
    assert set(system["backends"]) >= {"mapanything", "sam3", "sam3d"}
    assert system["can_reconstruct"] == bool(system["backends"]["mapanything"] and system["backends"]["sam3"])
    assert ".mp4" in system["accepted"]["videos"]
    assert system["default_prompts"]


def test_scene_is_listed(client, demo):
    scenes = client.get("/api/scenes").json()["scenes"]
    mine = next(s for s in scenes if s["id"] == demo)
    assert mine["name"] == "Test room"
    assert mine["source"] == "demo"
    assert mine["n_objects"] == 3


def test_unknown_ids_are_404(client, demo):
    assert client.get("/api/scenes/nope").status_code == 404
    assert client.get("/api/scenes/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/does-not-exist").status_code == 404
    res = client.post(f"/api/scenes/{demo}/query/dimensions", json={"object": "ghost_1"})
    assert res.status_code == 404


# --------------------------------------------------------------------------
# document and geometry
# --------------------------------------------------------------------------
def test_scene_document(client, demo):
    doc = client.get(f"/api/scenes/{demo}").json()
    assert sorted(objects_of(doc)) == IDS
    assert doc["units"] == "metres" and doc["up_axis"] == "+Z"
    assert doc["layout"]["floor_found"] and len(doc["layout"]["walls"]) == 2
    assert 0.0 < doc["scale"]["sigma_rel"] < 0.2
    assert len(doc["cameras"]) == 10
    assert all(c["has_frame"] and c["has_depth"] for c in doc["cameras"])
    assert doc["ground_truth"], "the synthetic room stores its exact sizes"

    table = objects_of(doc)["table_1"]
    for axis in ("length", "width", "height"):
        m = table["dimensions"][axis]
        assert m["sigma_m"] > 0
        assert m["interval95_m"][0] < m["value_m"] < m["interval95_m"][1]
    assert objects_of(doc)["box_1"]["support"] == "table_1"
    assert table["dependants"] == ["box_1"]


def test_document_matches_the_library(client, demo):
    """The API adds nothing of its own: numbers equal what scenelift.query returns."""
    doc = client.get(f"/api/scenes/{demo}").json()
    scene = Scene.load(client.app.state.store.path(demo))
    for oid, o in objects_of(doc).items():
        for axis, m in query.dimensions(scene, oid).items():
            got = o["dimensions"][axis]
            # the folder stores float32 arrays, so allow for that rounding
            assert got["value_m"] == pytest.approx(m.value, abs=2e-4)
            assert got["sigma_m"] == pytest.approx(m.sigma, abs=2e-4)
            assert got["relies_on_generated"] == m.relies_on_generated


def test_geometry_payload(client, demo):
    doc = client.get(f"/api/scenes/{demo}").json()
    geo = client.get(f"/api/scenes/{demo}/geometry").json()
    assert geo["revision"] == doc["revision"]
    assert sorted(geo["objects"]) == IDS
    for oid, g in geo["objects"].items():
        o = objects_of(doc)[oid]
        mesh = g["mesh"]
        vertices = np.frombuffer(base64.b64decode(mesh["vertices"]), "<f4").reshape(-1, 3)
        faces = np.frombuffer(base64.b64decode(mesh["faces"]), "<u4").reshape(-1, 3)
        colors = np.frombuffer(base64.b64decode(mesh["colors"]), "u1").reshape(-1, 3)
        observed = np.frombuffer(base64.b64decode(mesh["observed"]), "u1")
        assert len(vertices) == mesh["n_vertices"] == o["n_vertices"]
        assert len(faces) == mesh["n_faces"] == o["n_faces"]
        assert len(colors) == len(observed) == len(vertices)
        assert faces.max() < len(vertices)
        assert set(np.unique(observed)) <= {0, 1}
        # the mesh lies inside the object's upright box (plus rounding)
        half = 0.5 * np.linalg.norm(o["obb"]["size"]) + 1e-3
        assert np.linalg.norm(vertices - np.array(o["obb"]["center"]), axis=1).max() <= half
        points = np.frombuffer(base64.b64decode(g["measured"]["points"]), "<f4").reshape(-1, 3)
        assert len(points) == g["measured"]["n_points"] > 0
    assert geo["background"]["n_points"] > 0


def test_frames(client, demo):
    res = client.get(f"/api/scenes/{demo}/frames/0")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.content[:2] == b"\xff\xd8"
    assert client.get(f"/api/scenes/{demo}/frames/99").status_code == 404


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------
def test_distance(client, demo):
    res = client.post(f"/api/scenes/{demo}/query/distance", json={"a": "table_1", "b": "cabinet_1"})
    assert res.status_code == 200, res.text
    m = res.json()
    assert m["value_m"] > 0 and m["sigma_m"] > 0
    a, b = (np.array(p) for p in m["segment"])
    assert np.linalg.norm(a - b) == pytest.approx(m["value_m"], abs=0.02)
    same = client.post(f"/api/scenes/{demo}/query/distance", json={"a": "table_1", "b": "table_1"})
    assert same.status_code == 422


def test_distance_to_wall(client, demo):
    m = client.post(f"/api/scenes/{demo}/query/distance-to-wall", json={"object": "cabinet_1"}).json()
    assert m["value_m"] >= 0 and m["sigma_m"] > 0
    assert len(m["segment"]) == 2


def test_fits_in_space(client, demo):
    url = f"/api/scenes/{demo}/query/fits-in-space"
    yes = client.post(url, json={"object": "box_1", "space": [0.4, 0.4, 0.4]}).json()
    no = client.post(url, json={"object": "cabinet_1", "space": [0.3, 0.3, 1.2]}).json()
    assert yes["verdict"] == "yes" and yes["probability"] > 0.95
    assert no["verdict"] == "no" and no["probability"] < 0.05
    assert set(yes["margins"]) == {"length", "width", "height"}


def test_fits_between(client, demo):
    url = f"/api/scenes/{demo}/query/fits-between"
    res = client.post(url, json={"object": "box_1", "a": "table_1", "b": "cabinet_1"})
    assert res.status_code == 200, res.text
    fit = res.json()
    assert fit["verdict"] in ("yes", "no", "uncertain")
    assert 0.0 <= fit["probability"] <= 1.0
    assert fit["gap"]["value_m"] > 0 and len(fit["segment"]) == 2
    twice = client.post(url, json={"object": "box_1", "a": "box_1", "b": "cabinet_1"})
    assert twice.status_code == 422


def test_placements(client, demo):
    res = client.post(f"/api/scenes/{demo}/query/placements", json={"object": "box_1"})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["placements"], "the room has free floor for a small box"
    assert len(out["footprint"]) == 2
    assert all(len(p["center_xy"]) == 2 for p in out["placements"])
    assert client.post(f"/api/scenes/{demo}/query/placements", json={}).status_code == 422


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def test_select_by_text(client, demo):
    res = client.post(f"/api/scenes/{demo}/select/text", json={"text": "cabinet"})
    assert res.json()["objects"] == ["cabinet_1"]


def test_select_by_click_and_box(client, demo):
    """Project an object's centre into a camera that sees it, then click there."""
    doc = client.get(f"/api/scenes/{demo}").json()
    centre = np.array(objects_of(doc)["cabinet_1"]["obb"]["center"])
    hits = []
    for cam in doc["cameras"]:
        K, cam2world = np.array(cam["K"]), np.array(cam["cam2world"])
        p = np.linalg.inv(cam2world) @ np.append(centre, 1.0)
        if p[2] <= 0:
            continue
        u, v = (K @ p[:3])[:2] / p[2]
        H, W = cam["size"]
        if not (0 <= u < W and 0 <= v < H):
            continue
        res = client.post(f"/api/scenes/{demo}/select/click", json={"camera": cam["index"], "u": u, "v": v})
        assert res.status_code == 200, res.text
        hits.append(res.json()["object"])
        if hits[-1] == "cabinet_1":
            box = client.post(f"/api/scenes/{demo}/select/box",
                              json={"camera": cam["index"], "box": [u + 6, v + 6, u - 6, v - 6]})
            assert "cabinet_1" in box.json()["objects"]
            break
    assert "cabinet_1" in hits, f"no camera saw the cabinet's centre unoccluded: {hits}"
    bad = client.post(f"/api/scenes/{demo}/select/click", json={"camera": 99, "u": 1, "v": 1})
    assert bad.status_code == 404


# --------------------------------------------------------------------------
# tools and evaluation
# --------------------------------------------------------------------------
def test_tools(client, demo):
    tools = client.get("/api/tools").json()["tools"]
    assert tools == query.TOOLS
    res = client.post(f"/api/scenes/{demo}/tools/call",
                      json={"name": "get_dimensions", "arguments": {"object": "table"}})
    assert res.status_code == 200, res.text
    assert res.json()["result"]
    missing = client.post(f"/api/scenes/{demo}/tools/call", json={"name": "get_dimensions", "arguments": {}})
    assert missing.status_code == 422
    unknown = client.post(f"/api/scenes/{demo}/tools/call", json={"name": "teleport", "arguments": {}})
    assert unknown.status_code == 404


def test_evaluate_against_stored_ground_truth(client, demo):
    truth = client.get(f"/api/scenes/{demo}").json()["ground_truth"]
    res = client.post(f"/api/scenes/{demo}/evaluate", json={"ground_truth": truth})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["n_lengths"] == 10 and out["n_fits"] == 2
    assert out["abs_rel_error"] < 0.08
    assert out["fit_accuracy"] == 1.0
    bad = client.post(f"/api/scenes/{demo}/evaluate", json={"ground_truth": [{"type": "dimension"}]})
    assert bad.status_code == 422


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt, magic", [("glb", b"glTF"), ("ply", b"ply"), ("points", b"ply")])
def test_export(client, demo, fmt, magic):
    res = client.get(f"/api/scenes/{demo}/export", params={"format": fmt})
    assert res.status_code == 200, res.text
    assert res.content.startswith(magic)
    assert "attachment" in res.headers["content-disposition"]


def test_export_obj(client, demo):
    res = client.get(f"/api/scenes/{demo}/export", params={"format": "obj"})
    assert res.status_code == 200, res.text
    assert len(res.content) > 100


def test_export_scene_folder(client, demo):
    res = client.get(f"/api/scenes/{demo}/export", params={"format": "zip"})
    names = zipfile.ZipFile(io.BytesIO(res.content)).namelist()
    assert "scene.json" in names
    assert any(n.startswith("frames/") for n in names)
    assert client.get(f"/api/scenes/{demo}/export", params={"format": "exe"}).status_code == 422


# --------------------------------------------------------------------------
# editing
# --------------------------------------------------------------------------
def test_translate_carries_what_rests_on_it(client, scratch):
    before = objects_of(client.get(f"/api/scenes/{scratch}").json())
    res = client.post(f"/api/scenes/{scratch}/edit",
                      json={"op": "translate", "object": "table_1", "delta": [0.3, 0.0, 0.0]})
    assert res.status_code == 200, res.text
    doc = res.json()["scene"]
    after = objects_of(doc)
    assert doc["revision"] > 0
    for oid in ("table_1", "box_1"):
        moved = np.array(after[oid]["obb"]["center"]) - np.array(before[oid]["obb"]["center"])
        assert moved == pytest.approx([0.3, 0.0, 0.0], abs=1e-3)
    still = np.array(after["cabinet_1"]["obb"]["center"]) - np.array(before["cabinet_1"]["obb"]["center"])
    assert still == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert doc["history"][-1]["op"] == "translate"
    # the geometry was rebuilt for the new revision
    assert client.get(f"/api/scenes/{scratch}/geometry").json()["revision"] == doc["revision"]


def test_rotate_and_duplicate(client, scratch):
    before = objects_of(client.get(f"/api/scenes/{scratch}").json())["cabinet_1"]["obb"]["yaw"]
    doc = client.post(f"/api/scenes/{scratch}/edit",
                      json={"op": "rotate", "object": "cabinet_1", "yaw_deg": 30}).json()["scene"]
    turned = objects_of(doc)["cabinet_1"]["obb"]["yaw"] - before
    # an upright box looks the same after half a turn, so compare the angles modulo that
    assert abs(((np.degrees(turned) + 45) % 90) - 45 - 30) < 1.0

    res = client.post(f"/api/scenes/{scratch}/edit",
                      json={"op": "duplicate", "object": "cabinet_1", "offset": [0.0, 0.9, 0.0]}).json()
    assert res["new_id"] in objects_of(res["scene"])
    assert len(res["scene"]["objects"]) == 4


def test_remove_drops_the_load(client, scratch):
    doc = client.post(f"/api/scenes/{scratch}/edit", json={"op": "remove", "object": "table_1"}).json()["scene"]
    objects = objects_of(doc)
    assert "table_1" not in objects
    box = objects["box_1"]
    assert box["support"] == "floor"
    assert box["obb"]["center"][2] - box["obb"]["size"][2] / 2 == pytest.approx(0.0, abs=0.02)


def test_drop(client, scratch):
    client.post(f"/api/scenes/{scratch}/edit",
                json={"op": "translate", "object": "cabinet_1", "delta": [0.0, 0.0, 0.4], "settle": False})
    res = client.post(f"/api/scenes/{scratch}/edit", json={"op": "drop", "object": "cabinet_1"}).json()
    assert res["support"] == "floor"
    cabinet = objects_of(res["scene"])["cabinet_1"]
    assert cabinet["obb"]["center"][2] - cabinet["obb"]["size"][2] / 2 == pytest.approx(0.0, abs=0.02)


def test_bad_edits_are_rejected(client, scratch):
    url = f"/api/scenes/{scratch}/edit"
    assert client.post(url, json={"op": "translate", "object": "table_1"}).status_code == 422
    assert client.post(url, json={"op": "rotate", "object": "table_1"}).status_code == 422
    assert client.post(url, json={"op": "remove"}).status_code == 422
    assert client.post(url, json={"op": "explode", "object": "table_1"}).status_code == 422
    assert client.post(url, json={"op": "remove", "object": "ghost_1"}).status_code == 404


def test_calibrate(client, scratch):
    before = client.get(f"/api/scenes/{scratch}").json()
    height = objects_of(before)["cabinet_1"]["dimensions"]["height"]["value_m"]
    res = client.post(f"/api/scenes/{scratch}/calibrate",
                      json={"object": "cabinet_1", "axis": "height", "true_m": 2 * height})
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["factor"] == pytest.approx(2.0, rel=1e-3)
    after = out["scene"]
    assert after["scale"]["calibrated"]
    assert objects_of(after)["cabinet_1"]["dimensions"]["height"]["value_m"] == pytest.approx(2 * height, rel=1e-3)
    assert after["scale"]["sigma_rel"] < before["scale"]["sigma_rel"]
    assert client.post(f"/api/scenes/{scratch}/calibrate", json={"true_m": 1.0}).status_code == 422


def test_edits_survive_a_restart(client, scratch):
    client.post(f"/api/scenes/{scratch}/edit", json={"op": "remove", "object": "cabinet_1"})
    # a second application over the same folder sees the edited scene
    settings = StudioSettings(scenes_dir=client.app.state.store.root, open_browser=False)
    with TestClient(create_app(settings)) as other:
        doc = other.get(f"/api/scenes/{scratch}").json()
    assert sorted(objects_of(doc)) == ["box_1", "table_1"]


# --------------------------------------------------------------------------
# the generator can be wrong, and absent
# --------------------------------------------------------------------------
def test_wrong_shape_is_rejected(client):
    scene_id = build_demo(client, name="Wrong shape", completer="wrong")
    doc = client.get(f"/api/scenes/{scene_id}").json()
    for o in doc["objects"]:
        assert o["completion"] == "hull"
        assert "completion_rejected" in o["flags"]
    client.delete(f"/api/scenes/{scene_id}")


def test_no_generator(client):
    scene_id = build_demo(client, name="No generator", completer="none")
    doc = client.get(f"/api/scenes/{scene_id}").json()
    assert all(o["completion"] == "hull" for o in doc["objects"])
    assert client.delete(f"/api/scenes/{scene_id}").status_code == 204
    assert client.get(f"/api/scenes/{scene_id}").status_code == 404


def test_demo_request_is_validated(client):
    assert client.post("/api/scenes/demo", json={"n_views": 1}).status_code == 422
    assert client.post("/api/scenes/demo", json={"completer": "magic"}).status_code == 422


# --------------------------------------------------------------------------
# uploads and the store
# --------------------------------------------------------------------------
def test_reconstruct_guards(client):
    """Without the model backends an upload is refused with a clear message;
    with them, a file of the wrong type is."""
    files = {"files": ("notes.txt", b"hello", "text/plain")}
    res = client.post("/api/scenes/reconstruct", files=files, data={"name": "x"})
    if client.get("/api/system").json()["can_reconstruct"]:
        assert res.status_code == 415
    else:
        assert res.status_code == 503
        assert "backends" in res.json()["detail"]


def test_store_ids(tmp_path):
    store = SceneStore(tmp_path)
    assert slugify("My Living Room!") == "my-living-room"
    first, second = store.new_id("Room"), store.new_id("Room")
    assert first != second
    for bad in ("../x", "a/b", "", ".hidden", "x" * 80):
        with pytest.raises(KeyError):
            store.path(bad)


def test_front_end_is_served(client):
    res = client.get("/")
    assert res.status_code in (200, 503)  # 503 only when the bundle was never built
    assert "SceneLift" in res.text
    assert client.get("/some/deep/link").status_code == res.status_code
