# scenelift: design document

Version 0.1, 13 November 2026

Author: Prof. Shahab Anbarjafari (shb@3sholding.com). Copyright 2026 3S Holding OÜ. Licensed under Apache 2.0.

Tagline: Film it. Edit it in 3D.

## 0. Read this first: what exists today and what does not

This document describes the full method. The repository that comes with it is version 0.1. To avoid any misunderstanding:

Implemented, with automated tests in `tests/test_synthetic.py` (run with `pytest -q`):

- The complete geometry core: frame selection, cross-view scale harmonisation, floor and wall detection, multi-view object association, fusion, registration of generated shapes to measured points, outline-based size refinement, observed versus generated tagging, support and gravity reasoning, editing, spatial queries with error bars, selection by click, box and text, save, load, glTF export, command line, evaluation.
- These are tested end to end on a synthetic room with exact ground truth, with injected per-view scale errors of about 5 percent and depth noise.

Implemented, with automated tests in `tests/test_studio.py`:

- SceneLift Studio: a web application started with `python app.py`. It is a REST API over the library (`scenelift/studio/`, documented in `docs/API.md`) and a React front end (`frontend/`, compiled into `scenelift/studio/static`). It builds the synthetic room on a laptop CPU, shows the scene in 3D with observed and generated geometry in different colours, and exposes every edit, query and selection of this document. This is the first version of the interactive viewer of roadmap item M4.2. Real footage in Studio goes through the same backend wrappers as the command line, so the note below applies to it as well.

Written but not yet run on real model weights:

- The three wrappers in `scenelift/backends/` for MapAnything, SAM 3 and SAM 3D Objects. They follow the usage shown in each project's README and demo code, read on 13 November 2026. They were written in an environment with no GPU and no weights, so expect small fixes on first contact. `tests/test_backends_live.py` is the smoke test for that day.

Not built yet (see the roadmap in section 12):

- Any trained component of our own, the real-photo benchmark, texture baking, background inpainting behind removed objects, Gaussian splat export, dragging objects with the mouse in the viewer (Studio moves them in steps), the hosted demo.

The synthetic test proves the geometry and the bookkeeping are right. It does not prove quality on real rooms. That is milestone M0.

## 1. The problem

Computer vision is strong at 2D recognition and segmentation. It is still weak at turning ordinary photos into a 3D understanding that a person or a program can use.

The target: given a short casual phone video, or a few photos, of a real scene, produce a clean, metric, editable 3D representation of the scene and its objects, plus reliable spatial answers, with no careful capture, no special hardware and no manual clean-up.

"Solved" means all of the following at once:

1. Metric scale. Sizes and distances in real metres.
2. Object-level output. Each object is separate and has complete geometry.
3. Consistency across views. No floaters, no warped or doubled objects.
4. Physical plausibility. Objects rest on something, do not float and do not sink.
5. Promptable. Click, box or text selects an object and returns its model and pose.
6. Spatial answers. "How far is the chair from the table?", "Will this fit there?".
7. Runs on consumer hardware, or has a clear cloud option.
8. Works on real, cluttered, badly lit, partly occluded scenes.

## 2. Where current tools stop

Each requirement has a strong public model. None covers the combination.

| Family | Examples | Gives | Missing |
|---|---|---|---|
| Feed-forward metric geometry | MapAnything, Depth Anything 3, VGGT line | Depth, cameras, points in metres from 1 to N images | No objects. Scale drifts between views. No error bars. |
| Single-image object reconstruction | SAM 3D Objects, CAST, 3D-RE-GEN | Complete textured objects with a layout guess | One image only. Scale and placement are guesses. Unseen and seen parts are not told apart. |
| Physics-aware composition | SimuScene and related | Scenes that survive a simulator | Single image, research code, no measurement. |
| Spatial question answering | Vision-language models | Fluent answers | The numbers are guessed, not computed. |

The gaps scenelift targets:

- Gap A: few-view casual capture (2 to 30 frames) with object-level output.
- Gap B: a strict separation between what was measured and what was generated.
- Gap C: calibrated uncertainty on every length and every "fits" answer.
- Gap D: spatial answers computed on the reconstruction, exposed as tools for language models.
- Gap E: one package that installs with one command and exports to standard formats.

## 3. Design principles

1. Measure first, generate second. Pose and scale always come from multi-view measurement. A generative model is only allowed to fill in what no camera saw.
2. Never hide a guess. Every vertex is tagged observed or generated. Every answer carries a sigma and a flag when it depends on generated geometry.
3. Frozen backbones behind small interfaces. We do not train a foundation model. We train small pieces later, where the data shows a gap.
4. Simple geometry that can be explained. Upright boxes, planes, voxel overlap, ICP. Each step can be tested alone.
5. The core must run and be testable on a laptop with no GPU. Only the backbones need acceleration.

## 4. System overview

```
video / photos
   |
   1  ingest          sharp, well-spaced keyframes                    ingest.py
   |
   2  geometry        metric depth, intrinsics, pose per frame        backends/mapanything.py
   |
   3  scale           views agree on scale; scale sigma is measured   scale.py
   |
   4  align           Z up, floor at 0, walls square to the axes      align.py
   |
   5  segment         2D instance masks per frame                     backends/sam3.py
   |
   6  associate       masks linked across views into 3D instances     association.py
   |
   7  fuse            one clean measured point cloud per object       fusion.py
   |
   8  complete        generated shape, registered, checked, tagged    completion.py, backends/sam3d.py
   |
   9  physics         supports, gravity, overlaps                     physics.py
   |
   Scene              edit, query, select, save, export               scene.py, query.py, select.py
```

Conventions, used everywhere:

- Units are metres.
- Camera frame is OpenCV: +X right, +Y down, +Z forward. `cam2world` is 4x4.
- After step 4 the world is Z-up with the floor at z = 0.
- Pixel (u, v) is (column, row). Integer coordinates are pixel centres.
- Every tunable number lives in `scenelift/config.py`.

## 5. The method, step by step

Each step lists: goal, micro steps, key parameters, known failure modes.

### Step 1. Ingest (`ingest.py`)

Goal: 8 to 24 sharp frames that cover the scene with real viewpoint change.

1. Accept a video file, a folder, one image or a list of paths.
2. For photos, apply the EXIF orientation so portrait shots are upright.
3. For video, read every `frame_stride`-th frame (default 3).
4. Score sharpness as the variance of the Laplacian on a 480 pixel grey copy.
5. Split the timeline into `2 * max_frames` equal windows and keep the sharpest frame of each. This removes motion blur without losing coverage.
6. Walk through the kept frames in order. Track up to 300 corners from the last accepted frame with pyramidal Lucas-Kanade at 320 pixel width. Accept the frame only if the median flow is at least `min_flow_px` (default 12). This removes moments when the user stood still.
7. If more than `max_frames` remain, subsample evenly.
8. Resize so the longer side is at most `max_image_side` (default 1280).

Failure modes: very fast pans (everything blurred), videos shorter than 2 seconds, rolling-shutter wobble. Mitigation later: gyroscope metadata when present.

### Step 2. Metric geometry (`backends/mapanything.py`)

Goal: for each frame, z-depth in metres, intrinsics, camera pose, confidence, validity.

1. Write the frames to a temporary folder and load them with the backbone's own loader, so preprocessing matches training.
2. Run one feed-forward pass over all frames with mixed precision on CUDA.
3. Read per frame: `depth_z`, `intrinsics`, `camera_poses` (cam2world, OpenCV), `conf`, `mask`, and the resized image `img_no_norm`.
4. Squash confidence to the range 0 to 1 per view using the 2nd and 98th percentiles, so one threshold works across scenes.
5. Mark as invalid any pixel with non-finite or non-positive depth.

All later steps work at the backbone's output resolution. The interface is `GeometryBackend.infer(frames) -> list[View]`. Any model that returns these fields can replace MapAnything.

### Step 3. Scale harmonisation and scale uncertainty (`scale.py`)

Goal: remove view-to-view scale drift, and measure how unstable the scale was.

This is the first place where scenelift adds something the backbones do not give.

1. For each view, build the usable-pixel mask: valid, confidence at least `min_conf`, depth between 0.05 m and `max_depth`, and not on a depth edge. A pixel is on a depth edge if depth changes by more than `depth_edge_rel` (4 percent) inside its 3x3 neighbourhood.
2. Unproject up to 20,000 usable pixels per view to world points.
3. For view i, take the points of all other views, project them into view i, and read view i's own depth at those pixels.
4. Compute r = log(z_projected / depth_i) for each match. Keep only |r| below `scale_gate` (0.25). Larger ratios are occlusions, not scale errors.
5. If at least `scale_min_matches` remain, the scale error of view i is the median of r.
6. Subtract the median over views from all per-view values. This fixes the gauge: the correction never shifts the global scale.
7. Multiply each view's depth by exp(value). Repeat steps 2 to 7 once more.
8. Store `sigma_consistency`, the robust spread (1.4826 times MAD) of the per-view values from the first round.
9. The relative scale uncertainty used by all queries is

   sigma_rel = sqrt( sigma_backbone^2 + (1.25 * sigma_consistency / sqrt(N))^2 )

   The second term is the standard error of the median over N views. The first term, `backbone_scale_sigma` (default 4 percent), is a prior on the scale bias shared by all views. That bias cannot be observed from the images. It must be measured on the benchmark (section 11) and is removed entirely when the user supplies one known length (`Scene.calibrate`).

Verified on synthetic data: injected per-view errors with 5 percent spread are recovered with a residual spread below 0.6 percent.

Failure modes: views with almost no overlap give too few matches and keep their original scale. Scaling depth about each camera centre assumes the predicted camera translations are right.

### Step 4. Gravity alignment and room layout (`align.py`)

Goal: a clean frame. Without it, "upright box", "resting on" and "fits" have no meaning.

Floor:

1. Up prior: the mean of minus the camera Y axis over all views. People hold phones roughly upright.
2. Sample 15,000 usable points per view.
3. RANSAC planes whose normal is within `floor_max_tilt_deg` (35 degrees) of the prior.
4. Score each plane as inliers minus 4 times the number of points more than 3 thresholds below the plane. The penalty is what separates the floor from table tops (the floor lies below them) and from the ceiling.
5. Refit the winner by SVD on its inliers.
6. Build the rigid transform that rotates the normal onto +Z and moves the plane to z = 0. Apply it to every camera pose.
7. If no plane holds at least 3 percent of the points, keep the prior direction, put z = 0 at the 1st percentile of height, and record `floor_found = False`.

Walls and yaw:

8. Take points above 15 cm. Run RANSAC for planes whose normal is within 8 degrees of horizontal, up to `max_walls` (4) times, removing inliers each time.
9. Keep a plane only if it holds at least `wall_min_fraction` of the points and its inliers span more than 1 m in height. The height test rejects the fronts of low furniture.
10. Take the largest wall. Rotate the scene about Z so its normal snaps to the nearest axis. Rooms then come out square to X and Y.
11. Orient every wall normal toward the mean camera position, so "in front of the wall" is always the room side.
12. Store the floor as a rectangle around all points and each wall as a quad from the floor to the 99th height percentile.

Failure modes: outdoor scenes without walls (fine, zero walls are stored), sloped ground, mirrors that create a false room behind the wall.

### Step 5. Segmentation (`backends/sam3.py`)

Goal: 2D instance masks per view, with labels and scores.

1. For each view, encode the image once.
2. Query it with every word in `Config.prompts` (37 household nouns by default, user-replaceable with `--prompts`).
3. Collect masks above the confidence threshold with the prompt as the label.

The interface is `SegmentationBackend.segment(views) -> list[list[Mask2D]]`. A `Mask2D` may carry a `track_id`. When the segmenter tracks objects through the video, step 6 trusts that identity first.

Failure modes: objects not covered by the prompt list are missed and end up in the background. One prompt per forward pass is slow. Roadmap item R3 moves to the video tracker.

### Step 6. Multi-view association (`association.py`)

Goal: decide which masks in different views are the same physical object.

1. Per view, drop masks under `mask_min_pixels` and remove duplicates: if two masks overlap with IoU above `mask_nms_iou`, keep the higher score.
2. Erode each mask by `mask_erode_px` (1 pixel) and intersect with the usable-pixel mask. Border pixels mix object and background depth and must never enter an object.
3. Unproject the mask to 3D points, with colour and confidence.
4. Quantise the points to voxels of `assoc_voxel` (5 cm) and pack each voxel index into one 64-bit key.
5. Match against existing instances with overlap = shared voxels divided by the smaller voxel count.
6. Required overlap is `assoc_overlap` (0.25). It rises to 0.5 if the labels differ and to 0.7 if one piece is less than a fifth of the other. This is the mug-on-table guard: a small object shares its contact voxels with a large one.
7. One instance may not take two masks from the same view.
8. No match: start a new instance. Match: union the voxels, add the observation, add a label vote weighted by score times area.
9. Merge pass, for pairs never seen in the same view. Merge if voxel overlap passes the rule of step 6. Otherwise merge if labels agree and the projection test passes in both directions: at least `assoc_projection` (70 percent) of one instance's points project inside the other's masks and lie no farther behind its visible surface than its own size. This is the wide-baseline case, where two photos see opposite faces that share no 3D points.
10. Drop instances seen in fewer than `min_views_per_object` views (2; relaxed to 1 for one or two input views).

Verified: no duplicates with 2 views 130 degrees apart.

Failure modes: two identical objects one behind the other along both viewing directions, and thin objects smaller than a voxel.

### Step 7. Fusion (`fusion.py`)

Goal: one clean measured cloud per object, and the background.

1. Concatenate all observations of an instance.
2. Voxel-average at `fuse_voxel` (1 cm), weighted by confidence.
3. Remove statistical outliers: a point goes if its mean distance to its 12 nearest neighbours exceeds median + 2 * max(MAD, 0.25 * median). The floor on the spread matters. On a clean surface the MAD is almost zero and a plain test eats the object's edges.
4. Keep the largest connected cluster (link radius 5 cm), but only if it holds at least 60 percent of the points, so a real object is never split.
5. Discard objects with fewer than `min_object_points`.
6. Background: in each view take usable pixels outside all kept masks dilated by 3 pixels, unproject, voxel-average at 2.5 cm.

### Step 8. Completion, registration and provenance (`completion.py`)

Goal: give each object a complete shape without lying about what was seen. This is the second place where scenelift adds something new.

1. Pick the best observation of the object: largest area times mean confidence, halved if the mask touches the image border.
2. Ask the completion backend for a mesh from that image and mask. The mesh arrives in the backend's own frame and scale. Both are ignored.
3. Sample the mesh surface.
4. Build 48 starting poses: the 24 axis-aligned rotations, each with and without a 45 degree turn about Z. For each, set the scale from the ratio of bounding-box diagonals and the translation from the box centres.
5. Coarse stage: 8 iterations of partial ICP on about 500 measured and 800 model points for each start. Keep the best 5.
6. Fine stage: 30 iterations on 2,500 points each.
7. Partial ICP matches one way only, measured point to nearest model point, trimmed to the best 85 percent, and solves a similarity transform (Umeyama) each iteration. The unseen back of the model is never asked to match anything. Total scale change is clamped to the range 0.5 to 2.
8. Free-space test: project the posed model into up to 8 views. A model point is a violation if it lies in front of the measured depth at its pixel by more than max(4 cm, 3 percent of depth). The camera saw through that location, so nothing can be there.
9. Rank the fine results by residual divided by object size plus violation fraction. Keep the best.
10. Outline refinement of the size. Measured points come from inside the masks, so they stop about a pixel short of the true edge, and a shape fitted to them alone comes out a few percent small. For 14 scales from 0.94 to 1.20: scale the model about its centre, re-fit rotation and translation with 10 rigid ICP iterations, project a dense sample (12 points per mask pixel) into up to 6 views, draw only the points not hidden behind a nearer measured surface, and compute IoU with the masks. Keep the scale with the best mean IoU. On the synthetic scene, this step together with the one-pixel erosion of step 6 and the outlier floor of step 7 cut the size error from 4 to 10 percent down to 1 to 3 percent.
11. Accept the generated shape only if all three hold: violation at most `max_freespace_violation`, outline IoU minus violation at least `min_fit_quality` (0.72), median residual at most max(3 cm, 5 percent of size). The thresholds were set on synthetic shapes and must be recalibrated on real data.
12. If rejected, or with no backend, use the convex hull of the measured points and add the flag `completion_rejected` where applicable.
13. Provenance: a vertex is observed if a measured point lies within `observed_tau` (2.5 cm), else generated. Observed vertices take the measured colour.
14. Store the fit diagnostics (residual, violation, outline IoU) on the object.

Verified: a correct shape given in a wrong pose and scale is recovered. A sphere offered for a box is rejected.

Failure modes: symmetric objects can come out turned by 90 or 180 degrees (harmless for geometry, wrong for texture). Thin or transparent objects have poor depth. Deformable objects seen in different states.

### Step 9. Physical plausibility (`physics.py`)

Goal: nothing floats, nothing sinks, and the scene knows what stands on what.

1. Give every object an upright box: minimum-area rectangle of its footprint, plus its height range.
2. Process objects from the lowest bottom upward, so a supporter is already placed when its load is examined.
3. Support candidates for object A: objects whose top is below A's centre and whose footprint contains at least 30 percent of a 5x5 grid of A's footprint. Pick the highest. If none, the floor.
4. Gap = A's bottom minus the support's top. If |gap| is within max(`support_tol`, 10 percent of A's height), move A vertically by minus the gap and record the support.
5. If the gap is larger and A is close to a wall plane, record support = "wall".
6. Otherwise record no support and add the flag `floating`.
7. For pairs not in a support relation whose height ranges overlap by more than 2 cm, run a separating-axis test on the footprints. An overlap up to `max_push` (4 cm) is resolved by pushing the smaller object sideways. A larger one is only flagged `overlaps:<id>`, because a chair tucked under a table overlaps its box legitimately.
8. After an edit: when an object is removed, whatever stood on it drops to the next surface below. When an object is moved or turned, whatever stands on it moves with it.

Limit: this is box-level reasoning. Mesh-level contact and stability under simulation are roadmap item R6.

### Step 10. Scene, editing and export (`scene.py`)

1. A `Scene` holds objects, layout, background points, cameras, `ScaleInfo`, the config and an edit history.
2. A `SceneObject` holds the measured points and colours, the optional mesh with per-vertex colours and observed flags, the completion type (none, hull, generated), the support, flags, view count, upright box, fit diagnostics and the accumulated edit transform.
3. Edits: `remove`, `translate`, `rotate`, `duplicate`, `calibrate`, `settle`.
4. `calibrate(measured, true)` rescales objects, background, layout, camera positions and stored depth about the origin, which keeps the floor at 0, and switches sigma_rel to the reference's own error.
5. `save(folder)` writes `scene.json` plus compressed arrays and, unless `Config.save_frames` is off, the keyframe of every camera as `frames/<index>.jpg`. `load(folder)` restores everything.
6. `export("x.glb")` writes one named node per object plus floor and wall quads, converted to Y-up as glTF requires. `.obj` and `.ply` are also supported. `export_points` writes all measured points.

### Step 11. Spatial queries and uncertainty (`query.py`)

Every answer is a `Measurement` (value, sigma, 95 percent interval, relies_on_generated) or a `FitResult` (probability, verdict, margins).

Uncertainty model for a length d:

    sigma^2 = (d * sigma_rel)^2 + 2 * sigma_local^2 + sum over generated end points of (k * h)^2

- sigma_rel comes from step 3.
- sigma_local (8 mm) is the surface noise at each end.
- For an end point on generated geometry, h is its distance to the nearest measured point of the same object and k is `generated_sigma_frac` (0.15).

The constants are stated openly and are meant to be fitted on the benchmark.

Queries:

1. `dimensions(id)`: length, width, height from the upright box. For each axis the unseen part is the full extent minus the measured extent, and it adds (k * unseen)^2 to the variance.
2. `distance(a, b, mode)`: surface mode samples 6,000 points on each object and takes the 5th smallest nearest-neighbour distance (robust to one stray sample). Centre mode uses the box centres.
3. `distance_to_wall(id)`.
4. `fits_in_space(id, (w, d, h))`: the space is exact (a tape measure or a product sheet). Probability = product over the three axes of Phi(margin / sigma). Both horizontal orientations are tried. Verdict: yes at or above 0.95, no at or below 0.05, otherwise uncertain.
5. `fits_between(id, a, b)`: both lengths come from the same reconstruction, so a scale error stretches them together and only acts on their difference. The margin's sigma uses (margin * sigma_rel)^2 plus the local terms. This is tighter than comparing with an outside number, and it is correct.
6. `find_placements(id)`: a top-down grid with 2 cm cells. A cell is usable only if the floor there was observed and nothing occupies it below the object's height. Unobserved floor counts as unknown, not free. Cells behind walls are blocked. The free mask is eroded by the object's footprint in both orientations, and the centre with the most clearance is returned for each free region.
7. `TOOLS` and `call_tool`: the same queries as JSON tool definitions for function calling. A language model translates the question into a call and never produces the number itself.

### Step 12. Prompting (`select.py`)

1. `by_click(scene, camera, u, v)`: splat all object samples and the background into a quarter-resolution image of that camera with a z-buffer, and return the object at the pixel (majority in a 3x3 window).
2. `by_box(scene, camera, box)`: objects covering at least 15 percent of the box, largest first.
3. `by_text(scene, text)`: label match. Attribute words such as colours belong in the segmenter prompts or in a language model on top of `TOOLS`.

## 6. Scene file format

```
my_scene/
  scene.json        units, up axis, objects (id, label, support, flags, box, fit, transform),
                    layout, scale info, cameras, config, edit history
  objects/<id>.npz  points, colors, mesh_vertices, mesh_faces, mesh_colors, vertex_observed
  background.npz    points, colors
  depth.npz         per-view depth as float16 (optional, used by click selection and re-checks)
  frames/0000.jpg   the keyframe of each camera (optional, used by viewers; Config.save_frames)
  studio.json       display name and origin (written by SceneLift Studio only)
```

The format is plain JSON plus NumPy arrays and JPEG keyframes so a web viewer or a game engine plug-in can read it without scenelift. `docs/SCENE_FORMAT.md` describes every key.

## 7. Backends and licences

scenelift's own code is Apache 2.0. Backends are separate installs and are never bundled.

| Role | Default | Code and weights licence | Hardware |
|---|---|---|---|
| Geometry | MapAnything, checkpoint `facebook/map-anything-apache` | Apache 2.0 code, Apache 2.0 checkpoint | CUDA recommended; Apple MPS and CPU to be tested |
| Segmentation | SAM 3 | SAM License (custom, not an OSI licence) | CUDA |
| Completion | SAM 3D Objects | SAM License (custom, not an OSI licence) | CUDA required by its code |

Points to act on:

1. The SAM License is a custom licence with use restrictions and trade-control clauses. Read it in full, ideally with legal advice, before depending on it for anything commercial.
2. For adoption, the default install should work with permissively licensed parts only. Action: evaluate an Apache or MIT segmenter and an MIT image-to-3D model (candidates to verify: TRELLIS, TripoSR) behind the same interfaces, and make the SAM backends an opt-in extra.
3. The other MapAnything checkpoint, `facebook/map-anything`, is not the Apache one. Read its model card before use.
4. Laptop mode: geometry on MPS, segmentation on MPS or CPU, no generative completion (convex hulls, clearly flagged). Full mode: one CUDA GPU or a cloud endpoint.

## 8. Repository layout

```
scenelift/
  __init__.py        public API
  config.py          every tunable number
  types.py           Frame, View, Mask2D, CompletedShape, Instance3D
  geometry.py        cameras, voxels, planes, Umeyama, partial ICP, upright boxes, SAT
  ingest.py          step 1
  scale.py           step 3, ScaleInfo
  align.py           step 4, Layout
  association.py     step 6
  fusion.py          step 7
  completion.py      step 8
  physics.py         step 9
  scene.py           Scene, SceneObject, editing, save, load, export
  query.py           step 11, TOOLS
  select.py          step 12
  pipeline.py        SceneLift: runs steps 1 to 9
  eval.py            accuracy and calibration metrics
  cli.py             command line
  backends/
    base.py          the three interfaces
    mapanything.py   geometry backend
    sam3.py          segmentation backend
    sam3d.py         completion backend
    synthetic.py     synthetic room and fake backends for tests
  studio/            SceneLift Studio, the web application
    app.py           REST API: every route is a thin wrapper over the library
    runners.py       the two ways to make a scene: synthetic room, real footage
    jobs.py          background jobs with per-stage progress
    store.py         the scene library folder, with a small in-memory cache
    serialize.py     scene document and geometry payload for the browser
    schemas.py       request bodies
    settings.py      host, port, scene folder, device
    server.py        command-line entry and uvicorn
    static/          the compiled front end (build output, committed)
app.py               python app.py: starts SceneLift Studio
frontend/            React + TypeScript + three.js source of the front end
static/img/logo.png  3S Holding logo
tests/
  test_synthetic.py      end-to-end tests with ground truth (no GPU)
  test_studio.py         tests of the Studio REST API on the synthetic room (no GPU)
  test_backends_live.py  smoke test for real backends (skipped unless installed)
docs/                DESIGN.md (this file), API.md, SCENE_FORMAT.md, BACKENDS.md
examples/            short scripts: quick start, tools for language models, ground truth
requirements.txt     the one requirements file
pyproject.toml, README.md, LICENSE, NOTICE, CHANGELOG.md, CONTRIBUTING.md
```

## 9. Install and run

```bash
pip install -r requirements.txt             # the one requirements file
python app.py                               # SceneLift Studio at http://localhost:8000
pytest -q                                   # the test suite, on a laptop CPU

pip install -e .                            # optional: puts the `scenelift` command on the PATH

# real footage (after installing the backends, section 7 and docs/BACKENDS.md)
scenelift run room.mp4 -o room --export room.glb
scenelift run room.mp4 -o room --sam3d-repo /path/to/sam-3d-objects
scenelift info room
scenelift size room sofa_1
scenelift measure room sofa_1 table_1
scenelift fit room sofa_1 --space 2.2 1.0 0.9
scenelift fit room chair_1 --between table_1 cabinet_1
scenelift place room chair_1
scenelift edit room remove table_1
scenelift calibrate room door_1 height 2.03
scenelift eval room ground_truth.json
```

```python
from scenelift import SceneLift, query, select

scene = SceneLift.default().run("room.mp4")
scene.summary()
query.distance(scene, "sofa_1", "table_1")
query.fits_in_space(scene, "sofa_1", (2.2, 1.0, 0.9))
select.by_click(scene, camera_index=3, u=412, v=260)
scene.translate("sofa_1", (0.5, 0, 0))
scene.remove("table_1")
scene.save("room"); scene.export("room.glb")
```

## 10. What the tests show today

Synthetic room: a table, a small box on the table, a cabinet turned 20 degrees, two walls, a floor. Ten views at 320 x 240 on a 130 degree arc. Each view's depth is multiplied by a random factor (5 percent spread) and given 0.3 percent pixel noise. The completer returns the right shape in a wrong pose and scale. The segmenter gives perfect masks without track ids.

| Check | Result |
|---|---|
| Per-view scale errors recovered | residual spread below 0.6 percent |
| Floor at z = 0, walls square to the axes | within 1 cm, exact axes |
| Object sizes | within 1 to 3 percent with 10 views at low noise, within 5 percent with 3 to 10 views at double noise, within 12 percent with 2 wide-baseline views |
| Table to cabinet gap | within 5 cm of 0.85 m, truth inside the 95 percent interval |
| Supports | box on table, table and cabinet on floor |
| Edits | box rides with the table, drops to the floor when the table is removed |
| Wrong shape from the completer | rejected, hull used, flag set |
| Two views 130 degrees apart | three objects, no duplicates |
| Click, box, text selection; save, load, glTF export; CLI; tools; eval | pass |

What this does not show: behaviour with real depth errors (smeared edges, non-rigid drift, reflective and transparent surfaces), real segmentation errors, and real generated shapes that are only approximately right.

## 11. Benchmark: scenelift-bench

Purpose: prove the claims and tune the constants. This is also a contribution on its own, because no public set pairs casual phone video with tape-measured distances and fit questions.

Protocol:

1. 100 scenes: 60 indoor rooms, 20 table tops, 20 outdoor.
2. Per scene: one 15 to 30 second phone video, and three photo subsets (2, 5 and 10 photos).
3. Per scene: at least 10 tape or laser measurements (object sizes, gaps between objects, object to wall) and 5 fit questions with known answers.
4. Ground truth stored in the JSON format read by `eval.py`.

Metrics (all implemented in `eval.py`):

1. Mean and median absolute relative error of lengths.
2. Coverage of the 68 and 95 percent intervals. Honest error bars give about 0.68 and 0.95.
3. Mean signed z score. Far from 0 means a shared scale bias, which sets `backbone_scale_sigma`.
4. Fit accuracy and Brier score.

Baselines to report against, on the same scenes:

1. Metric depth plus masks, single view, measured directly on the unprojected points.
2. MapAnything points plus masks, no harmonisation, no completion.
3. SAM 3D Objects layout from one image.
4. Gaussian splatting on the same video, measured by hand in a viewer.
5. A vision-language model asked the same questions directly.

## 12. Roadmap to a release people will star

M0. First real scenes (1 to 2 weeks)

1. Install the three backends on one CUDA machine.
2. Run `tests/test_backends_live.py` on 5 phone videos. Fix the wrappers.
3. Save every intermediate (masks, clouds, fits) and review by eye.
4. Record 10 tape measurements per scene and run `scenelift eval`.
5. Decision gate: is scale error below 8 percent and are objects separated in most scenes? If not, the first research task is whichever step failed most often.

M1. Quality on real scenes (3 to 4 weeks)

- R1. Snap observed regions of the generated mesh onto the measured surface with Laplacian or as-rigid-as-possible deformation, so measured detail is kept exactly.
- R2. Texture: bake multi-view colour onto observed faces, keep generated texture elsewhere.
- R3. Use the SAM 3 video tracker to set `track_id`, and add automatic prompt discovery with a tagging model.
- R4. Background: inpaint floor and wall texture behind removed objects.
- R5. Gaussian splat export per object for photoreal viewing, next to the mesh.
- R6. Mesh-level contact and a short physics-simulation check.
- R7. Detect reference objects of known size (doors, sockets, A4 paper, bank cards) and calibrate automatically.

M2. Trained components, the research core (6 to 8 weeks)

- T1. Scale head: a small network on frozen backbone features that predicts the per-view scale correction and its sigma, trained on LiDAR datasets such as ScanNet++ and ARKitScenes. Replaces the fixed prior of step 3.
- T2. Learned registration start: predict the pose of the generated shape from image and points, replacing the 48-start search.
- T3. Calibration layer: fit sigma_local, k and the acceptance thresholds on the benchmark so that 95 percent intervals cover 95 percent.

M3. Benchmark and paper (in parallel with M1 and M2)

- Collect scenelift-bench, run the baselines, write the paper around three claims: measured versus generated separation, calibrated spatial answers, few-view object-level reconstruction.

M4. Packaging (2 weeks)

1. `pip install scenelift` with a permissive default stack.
2. One-click demo: upload a video, get an interactive scene in the browser, click to delete and measure. The local version exists: SceneLift Studio, `python app.py`. Still open: dragging objects with the mouse, and proof on real footage (M0).
3. Blender add-on and USD export.
4. A hosted demo and a Colab notebook.
5. Model card, licence table, limitations page.

M5. Launch

1. A 30 second clip: phone video on the left, the same room on the right with furniture being dragged, a measurement appearing, an object deleted.
2. Side-by-side comparisons against the baselines on the same scenes.
3. README with the clip at the top and a three-line quick start.
4. Post to the usual channels on one day, with the paper, the demo and the benchmark live at the same time.

## 13. Known limitations

1. Static scenes only. Moving people and pets are not handled.
2. Transparent, mirror-like and very thin objects get poor depth from every current backbone.
3. Objects missing from the prompt list are not separated.
4. Upright boxes are a coarse stand-in for shape in the physics and fit tests. L-shaped sofas and open shelves are over-approximated.
5. The absolute scale has a shared error that only a reference length or the trained scale head (T1) can remove.
6. All acceptance thresholds and uncertainty constants were set on synthetic data.
7. Outdoor scenes with no floor plane and no walls fall back to the gravity prior.
