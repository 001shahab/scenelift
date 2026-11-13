# SceneLift Studio REST API

`python app.py` serves the browser UI and this API from the same port. The
API is a thin wrapper: reconstruction is `scenelift.pipeline`, edits are
methods of `scenelift.scene.Scene`, answers come from `scenelift.query`, and
selections from `scenelift.select`. Anything you can do in the browser you can
do with `curl`.

- Base URL: `http://127.0.0.1:8000`
- Interactive documentation, generated from the code: `/api/docs`
- OpenAPI schema: `/api/openapi.json`

## Conventions

- Lengths are **metres**. Angles are **degrees**, except `obb.yaw`, which is radians.
- The world is **Z-up** with the floor at `z = 0`.
- Cameras are OpenCV: `+X` right, `+Y` down, `+Z` forward. `cam2world` is 4x4.
- A pixel `(u, v)` is `(column, row)` in the camera's own image size
  (`cameras[i].size` is `[height, width]`).
- Request and response bodies are JSON, except the upload (multipart form) and
  the downloads.

A length is always a **measurement**:

```json
{
  "what": "gap between table_1 and cabinet_1",
  "value_m": 0.8471,
  "sigma_m": 0.0362,
  "interval95_m": [0.7761, 0.9181],
  "relies_on_generated": false
}
```

A fit answer is always a **fit result**:

```json
{
  "verdict": "yes",
  "probability": 0.9991,
  "orientation": "as scanned",
  "relies_on_generated": false,
  "margins": { "length": { "what": "spare room", "value_m": 0.148, "sigma_m": 0.016 }, "width": {}, "height": {} }
}
```

`verdict` is `yes` at `probability >= Config.fit_yes` (0.95), `no` at
`<= Config.fit_no` (0.05), otherwise `uncertain`. `orientation` is
`as scanned` or `turned 90 degrees` for a space, and `width across the gap` or
`length across the gap` between two objects. Each margin is a measurement of
the spare room on that axis; a negative value means it is short by that much.

The numbers in the examples of this page show the format. They are not
benchmark results.

## Errors

Errors are JSON: `{"detail": "..."}`.

| Status | Meaning |
|---|---|
| 400 | The library rejected the input (`ValueError`) |
| 404 | Unknown scene, object, job, camera or tool |
| 413 | The upload is larger than the limit |
| 415 | Unsupported file type in an upload |
| 422 | The request body is malformed or incomplete |
| 503 | Real footage was uploaded but the model backends are not installed |

## System

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status": "ok", "version": "0.1.0"}` |
| GET | `/api/system` | Version, the scene library folder, which model backends are installed, `can_reconstruct`, pipeline stage names, default prompts, accepted file types, the upload limit |

## Jobs

Making a scene takes time, so it runs in the background. `POST
/api/scenes/demo` and `POST /api/scenes/reconstruct` return a job with status
`202`; poll it until `status` is `done` or `error`. One job runs at a time,
the others wait in a queue.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/jobs` | `{"jobs": [...]}`, newest first |
| GET | `/api/jobs/{job_id}` | One job |

```json
{
  "id": "3f9c1a2b7d10",
  "kind": "demo",
  "name": "Synthetic room",
  "status": "running",
  "error": null,
  "scene_id": null,
  "created_at": "2026-11-13T10:00:00+00:00",
  "finished_at": null,
  "stages": [
    { "name": "ingest", "status": "done", "seconds": 0.412 },
    { "name": "geometry", "status": "running", "seconds": 0.105 },
    { "name": "scale", "status": "pending", "seconds": null }
  ]
}
```

`status` is `queued`, `running`, `done` or `error`. When it is `done`,
`scene_id` names the new scene.

## Scenes

| Method | Path | Does |
|---|---|---|
| GET | `/api/scenes` | Lists the scenes in the library folder |
| POST | `/api/scenes/demo` | Builds the synthetic room (job) |
| POST | `/api/scenes/reconstruct` | Reconstructs real footage (job) |
| GET | `/api/scenes/{id}` | The scene document |
| GET | `/api/scenes/{id}/geometry` | Meshes and points as typed arrays |
| GET | `/api/scenes/{id}/frames/{index}` | One keyframe as an image |
| GET | `/api/scenes/{id}/export` | Download as glb, obj, ply, points or zip |
| DELETE | `/api/scenes/{id}` | Deletes the scene folder (status 204) |

### POST `/api/scenes/demo`

Runs the complete geometry core on a synthetic room with exact ground truth.
No GPU, no model weights.

| Field | Default | Meaning |
|---|---|---|
| `name` | `"Synthetic room"` | Display name |
| `n_views` | `10` | Cameras on a 130 degree arc, 2 to 24 |
| `scale_jitter` | `0.05` | Spread of the scale error injected into each view |
| `depth_noise` | `0.003` | Relative pixel noise on depth |
| `seed` | `0` | Random seed of the room |
| `completer` | `"correct"` | `correct`: the right shape in a wrong pose and scale. `wrong`: a sphere for every object, which must be rejected. `none`: no generator, convex hulls only |

```bash
curl -X POST localhost:8000/api/scenes/demo \
     -H 'Content-Type: application/json' \
     -d '{"name": "My first room", "n_views": 10}'
```

### POST `/api/scenes/reconstruct`

Multipart form. Needs MapAnything and SAM 3 on the server, otherwise `503`.

| Field | Meaning |
|---|---|
| `files` | One video, or one or more photos. Repeat the field for several photos |
| `name` | Display name |
| `max_frames` | Keyframes kept, 1 to 64. Default `Config.max_frames` (24) |
| `prompts` | Comma-separated object names to look for. Default: the household nouns of `Config.prompts` |
| `completion` | `true` (default) uses SAM 3D Objects when the server was started with `--sam3d-repo` |
| `config` | A JSON object that overrides any field of `scenelift.Config` |

```bash
curl -X POST localhost:8000/api/scenes/reconstruct \
     -F files=@room.mp4 -F name="Living room" -F max_frames=16 \
     -F prompts="sofa, table, chair, lamp"
```

### GET `/api/scenes/{id}`: the scene document

```json
{
  "id": "synthetic-room",
  "name": "Synthetic room",
  "source": "demo",
  "created_at": "2026-11-13T10:00:07+00:00",
  "inputs": ["synthetic room, 10 views"],
  "params": {},
  "ground_truth": [],
  "revision": 0,
  "units": "metres",
  "up_axis": "+Z",
  "objects": [
    {
      "id": "table_1",
      "label": "table",
      "completion": "generated",
      "support": "floor",
      "flags": [],
      "n_views": 10,
      "observed_fraction": 0.71,
      "fit": { "residual_m": 0.004, "freespace_violation": 0.0, "outline_iou": 0.93 },
      "obb": { "center": [0.0, 0.0, 0.375], "size": [1.2, 0.7, 0.75], "yaw": 0.0 },
      "dimensions": { "length": {}, "width": {}, "height": {} },
      "n_points": 9120,
      "n_vertices": 2402,
      "n_faces": 4800,
      "dependants": ["box_1"]
    }
  ],
  "layout": { "floor_found": true, "floor_quad": [], "walls": [] },
  "scale": {
    "per_view": [],
    "sigma_consistency": 0.047,
    "sigma_backbone": 0.04,
    "calibrated": false,
    "sigma_reference": 0.0,
    "global_factor": 1.0,
    "n_views": 10,
    "sigma_rel": 0.044
  },
  "cameras": [
    { "index": 0, "K": [], "cam2world": [], "size": [240, 320], "source": "synthetic#0",
      "has_frame": true, "has_depth": true }
  ],
  "config": {},
  "timings": { "ingest": 0.4, "geometry": 0.1 },
  "history": [],
  "bounds": { "min": [-2.0, -2.0, 0.0], "max": [2.0, 2.0, 2.4] }
}
```

- `completion` is `generated` (a generated shape that passed the checks),
  `hull` (the convex hull of the measured points) or `none`.
- `support` is an object id, `"floor"`, `"wall"` or `null` (then `flags`
  contains `floating`).
- `flags` may contain `floating`, `overlaps:<id>` and `completion_rejected`.
- `observed_fraction` is the share of the surface that a camera measured.
- `dimensions` holds one measurement per axis.
- `revision` counts the edits made since the server started. Reload the
  geometry when it changes.
- `ground_truth` is present for the synthetic room only.

### GET `/api/scenes/{id}/geometry`

Arrays are base64 strings of little-endian raw bytes.

```json
{
  "revision": 0,
  "objects": {
    "table_1": {
      "measured": { "n_points": 9120, "points": "<f4, n x 3>", "colors": "<u1, n x 3>" },
      "mesh": {
        "n_vertices": 2402, "n_faces": 4800,
        "vertices": "<f4, n x 3>", "faces": "<u4, m x 3>",
        "colors": "<u1, n x 3>", "observed": "<u1, n>"
      }
    }
  },
  "background": { "n_points": 80000, "points": "<f4, n x 3>", "colors": "<u1, n x 3>" }
}
```

`observed[i]` is 1 when vertex `i` lies on measured geometry and 0 when it was
generated. `mesh` is `null` for an object without a surface. Large clouds are
thinned for display; the scene on disk is never changed.

```python
import base64, numpy as np, requests

geo = requests.get("http://localhost:8000/api/scenes/synthetic-room/geometry").json()
mesh = geo["objects"]["table_1"]["mesh"]
vertices = np.frombuffer(base64.b64decode(mesh["vertices"]), "<f4").reshape(-1, 3)
observed = np.frombuffer(base64.b64decode(mesh["observed"]), "u1").astype(bool)
```

### GET `/api/scenes/{id}/frames/{index}`

The keyframe of camera `index` as JPEG. A scene saved without keyframes but
with depth returns a colour-mapped depth image (PNG) instead, so click
selection still has something to click on.

### GET `/api/scenes/{id}/export`

| Query | Values |
|---|---|
| `format` | `glb` (default; Y-up as glTF requires), `obj`, `ply` (Z-up), `points` (all measured points as one PLY), `zip` (the scene folder) |
| `background` | `true` adds the background points to a `glb` |

```bash
curl -OJ "localhost:8000/api/scenes/synthetic-room/export?format=glb"
```

## Editing

### POST `/api/scenes/{id}/edit`

| `op` | Needs | Does |
|---|---|---|
| `remove` | `object` | Deletes the object. Whatever stood on it drops to the surface below |
| `translate` | `object`, `delta: [dx, dy, dz]` | Moves it |
| `rotate` | `object`, `yaw_deg` | Turns it about the vertical axis through its centre |
| `duplicate` | `object`, optional `offset: [dx, dy, dz]` | Copies it. The response has `new_id` |
| `drop` | `object` | Lowers it onto whatever is underneath. The response has `support` |
| `settle` | nothing | Re-runs the support and gravity reasoning of step 9 |

Optional fields: `carry` (default `true`) also moves whatever rests on the
object. `settle` re-runs the physics after the edit; by default that happens
for `remove`, `drop` and `settle` only, as in the library.

The response is `{"scene": <scene document>, ...}`. The edit is written to the
scene folder before the response is sent.

```bash
curl -X POST localhost:8000/api/scenes/synthetic-room/edit \
     -H 'Content-Type: application/json' \
     -d '{"op": "translate", "object": "table_1", "delta": [0.3, 0, 0]}'
```

### POST `/api/scenes/{id}/calibrate`

Fixes the absolute scale with one known length (`Scene.calibrate`).

```json
{ "object": "cabinet_1", "axis": "height", "true_m": 1.02 }
```

Or give the length as the scene reports it: `{"measured_m": 0.98, "true_m": 1.02}`.
Returns `{"factor": 1.0408, "scene": <scene document>}`.

## Spatial queries

All are `POST /api/scenes/{id}/query/...` with a JSON body.

| Path | Body | Returns |
|---|---|---|
| `dimensions` | `{"object"}` | `{"length", "width", "height"}`, each a measurement |
| `distance` | `{"a", "b", "mode": "surface" or "center"}` | A measurement plus `segment`: the two end points, for drawing |
| `distance-to-wall` | `{"object"}` | A measurement plus `segment` and `wall`, or `null` when the scene has no walls |
| `fits-in-space` | `{"object", "space": [w, d, h], "allow_rotation": true, "clearance": 0}` | A fit result |
| `fits-between` | `{"object", "a", "b", "clearance": 0}` | A fit result plus `gap` (a measurement) and `segment` |
| `placements` | `{"object", "clearance": 0.02, "max_results": 5}` or `{"footprint": [l, w], "height"}` | `{"placements": [{"center_xy", "yaw_deg", "spare_m"}], "footprint", "height"}` |

```bash
curl -X POST localhost:8000/api/scenes/synthetic-room/query/fits-in-space \
     -H 'Content-Type: application/json' \
     -d '{"object": "cabinet_1", "space": [0.6, 0.5, 1.1]}'
```

## Selection

| Path | Body | Returns |
|---|---|---|
| `POST /api/scenes/{id}/select/click` | `{"camera", "u", "v"}` | `{"object": "table_1"}` or `{"object": null}` for background |
| `POST /api/scenes/{id}/select/box` | `{"camera", "box": [u0, v0, u1, v1]}` | `{"objects": [...]}`, largest share of the box first |
| `POST /api/scenes/{id}/select/text` | `{"text"}` | `{"objects": [...]}` |

## Tools for language models

| Method | Path | Does |
|---|---|---|
| GET | `/api/tools` | `{"tools": [...]}`: the JSON tool definitions of `scenelift.query.TOOLS` |
| POST | `/api/scenes/{id}/tools/call` | Runs one tool: `{"name": "measure_distance", "arguments": {"a": "table", "b": "cabinet"}}` |

Object arguments may be ids (`table_1`) or names (`table`). An ambiguous name
is an error that lists the candidates, so a model can ask again.

## Evaluation

### POST `/api/scenes/{id}/evaluate`

```json
{
  "ground_truth": [
    { "type": "dimension", "object": "table_1", "axis": "length", "true_m": 1.20 },
    { "type": "distance", "a": "table_1", "b": "cabinet_1", "mode": "surface", "true_m": 0.85 },
    { "type": "fit", "object": "box_1", "space": [0.4, 0.4, 0.4], "true": true }
  ]
}
```

Returns `n_lengths`, `n_fits`, `abs_rel_error`, `median_rel_error`,
`coverage_68`, `coverage_95`, `mean_z`, `fit_accuracy`, `fit_brier` and one
row per entry. See [the README](../README.md#evaluate) for what each number means.

## Configuration of the server

| Option | Environment variable | Default |
|---|---|---|
| `--host` | `SCENELIFT_HOST` | `127.0.0.1` |
| `--port` | `SCENELIFT_PORT` | `8000` |
| `--scenes-dir` | `SCENELIFT_SCENES_DIR` | `./scenes` |
| `--device` | `SCENELIFT_DEVICE` | chosen by the backends: cuda, then mps, then cpu |
| `--sam3d-repo` | `SAM3D_REPO` | none: no generative completion |
| | `SCENELIFT_MAX_UPLOAD_MB` | `2048` |
| `--no-browser` | | a browser tab is opened |

The server has no authentication. Read [SECURITY.md](../SECURITY.md) before
binding it to anything but `127.0.0.1`.
