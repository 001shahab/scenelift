# The scene folder

A scene is a folder of plain JSON, NumPy arrays and JPEG images. A web viewer,
a game-engine plug-in or a notebook can read it without scenelift.

```
my_scene/
  scene.json          everything that is not a large array
  objects/<id>.npz    one file per object
  background.npz      measured points that belong to no object
  depth.npz           per-view depth (optional)
  frames/0000.jpg     the keyframe of camera 0 (optional), 0001.jpg, ...
  studio.json         display name and origin, written by SceneLift Studio only
```

Write it with `scene.save("my_scene")`, read it with `Scene.load("my_scene")`.

## Conventions

- Units are metres.
- The world is Z-up with the floor at `z = 0`. Walls are square to X and Y.
- Cameras are OpenCV: `+X` right, `+Y` down, `+Z` forward. `cam2world` is a
  4x4 matrix that maps camera coordinates to world coordinates.
- A pixel `(u, v)` is `(column, row)`. Integer coordinates are pixel centres.

## scene.json

| Key | Content |
|---|---|
| `format_version` | Version of this format |
| `units` | `"metres"` |
| `up_axis` | `"+Z"` |
| `objects` | One entry per object, see below |
| `layout` | `floor_found`, `floor_quad` (4 x 3), and `walls`: each with `normal` (3), `d` and `quad` (4 x 3). A point `p` is in front of a wall when `normal . p + d > 0`; normals point into the room |
| `scale` | `per_view` (the correction applied to each view's depth), `sigma_consistency`, `sigma_backbone`, `calibrated`, `sigma_reference`, `global_factor`, `n_views`, and the resulting `sigma_rel` |
| `cameras` | One entry per keyframe: `K` (3 x 3 intrinsics), `cam2world` (4 x 4), `size` (`[height, width]`), `source` (where the frame came from) and `frame` (path of the keyframe relative to the folder, or `null`) |
| `config` | `config`: every tunable number the scene was built with. `timings`: seconds per pipeline stage |
| `history` | The edits applied since reconstruction, in order |

Each object:

| Key | Content |
|---|---|
| `id` | Unique name, for example `table_1` |
| `label` | What it is, for example `table` |
| `completion` | `generated` (a generated shape that passed the checks), `hull` (convex hull of the measured points) or `none` |
| `support` | The id of the object it stands on, `"floor"`, `"wall"`, or `null` when it floats |
| `flags` | Any of `floating`, `overlaps:<id>`, `completion_rejected` |
| `n_views` | Number of views that saw it |
| `observed_fraction` | Share of the surface that was measured rather than generated, 0 to 1 |
| `obb` | The upright box: `center` (3), `size` (length, width, height; length is the longer horizontal side) and `yaw` in radians about Z |
| `fit` | Diagnostics of the completion: `residual_m`, `freespace_violation`, `outline_iou` |
| `transform` | 4 x 4: the edits accumulated since reconstruction |

## objects/&lt;id&gt;.npz

| Array | Shape and type | Content |
|---|---|---|
| `points` | n x 3, float32 | The measured points of the object |
| `colors` | n x 3, uint8 | Their colours |
| `mesh_vertices` | m x 3, float32 | The complete surface (absent when `completion` is `none`) |
| `mesh_faces` | k x 3, int32 | Triangles |
| `mesh_colors` | m x 3, uint8 | Per-vertex colour |
| `vertex_observed` | m, bool | `True` where the vertex lies on measured geometry, `False` where it was generated |

`vertex_observed` is the provenance record. Anything that reads the mesh can
tell which parts a camera saw.

## background.npz

`points` (n x 3, float32) and `colors` (n x 3, uint8): everything measured
that was not assigned to an object, voxel-averaged at 2.5 cm.

## depth.npz (optional)

One array per camera, named `d0`, `d1`, ..., float16, metres, at the camera's
`size`. Used by selection in a keyframe and by re-checks. Leave it out with
`Config.save_depth = False`.

## frames/ (optional)

`frames/<index>.jpg`, zero-padded to four digits, is the keyframe of camera
`index` at the camera's `size`. Viewers use it to show the photo an object was
selected in. A keyframe that is already on disk is copied when the scene is
saved again, never re-encoded. Leave the folder out with
`Config.save_frames = False`.

A scene folder from before keyframes were stored loads as before: `frame` is
then absent and viewers fall back to the depth image.

## Reading a scene without scenelift

```python
import json
import numpy as np

doc = json.load(open("my_scene/scene.json"))
for obj in doc["objects"]:
    arrays = np.load(f"my_scene/objects/{obj['id']}.npz")
    length, width, height = obj["obb"]["size"]
    seen = arrays["vertex_observed"].mean() if "vertex_observed" in arrays else 1.0
    print(f"{obj['id']}: {length:.2f} x {width:.2f} x {height:.2f} m, {seen:.0%} observed")
```

## Export formats

| Call | Result |
|---|---|
| `scene.export("room.glb")` | glTF binary with one named node per object plus floor and wall quads, converted to **Y-up** as glTF requires |
| `scene.export("room.obj")`, `scene.export("room.ply")` | Meshes, Z-up, metres |
| `scene.export_points("points.ply")` | Every measured point, objects and background, as one coloured PLY |
