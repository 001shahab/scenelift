# Changelog

All notable changes to scenelift are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-13

First public release.

### Added

- **Geometry core** (CPU only, no model weights): keyframe selection, cross-view
  scale harmonisation with a measured scale uncertainty, floor and wall
  detection, multi-view object association, fusion, registration of generated
  shapes to measured points, outline-based size refinement, observed versus
  generated tagging per vertex, and support and gravity reasoning.
- **Scene**: remove, translate, rotate, duplicate, calibrate and settle; save and
  load as plain JSON plus NumPy arrays; export to glTF, OBJ and PLY.
- **Spatial queries with error bars**: dimensions, distance, distance to wall,
  fits in a space, fits between two objects, free floor placements. The same
  queries are exposed as JSON tool definitions for language models.
- **Selection** by click, box and text.
- **Command line**: `scenelift run | info | size | measure | fit | place | edit |
  calibrate | export | eval | studio`.
- **Evaluation**: accuracy and calibration metrics against tape-measure ground truth.
- **Backend wrappers** for MapAnything, SAM 3 and SAM 3D Objects behind three
  small interfaces, plus a synthetic room with exact ground truth.
- **SceneLift Studio**: a web application started with `python app.py`. A REST
  API over the library and a React front end with a 3D view, provenance
  colouring, editing, measuring, fit checks, placements, keyframe click and box
  selection, the tool playground, evaluation and exports.
- Keyframes are stored with the scene (`frames/<index>.jpg`), so a viewer can
  show the photo an object was selected in. Turn it off with
  `Config.save_frames = False`.

- Screenshots of Studio in the README, taken from a running instance on the
  synthetic room (`docs/images/`), and three of the first real room.
- Live backend tests that need only MapAnything: the contract of the geometry
  backend, and real depth through the core without masks
  (`tests/test_backends_live.py`).
- SAM 3 runs through the implementation in Hugging Face `transformers`
  (`pip install transformers`), which is not tied to CUDA. Run on Apple
  silicon so far. Meta's own package needs Linux with CUDA.
- Association handles nested masks: a mask must account for a share of the
  object as the photo sees it, and the reverse (`Config.assoc_cover`). A
  cabinet and the things in it no longer swallow each other.
- The frames of one video share one set of camera intrinsics
  (`MapAnythingBackend(share_intrinsics=...)`).
- Studio: gold hairlines and short crimson strips as trim on the frame of the
  application, in both themes. The README screenshots predate it.

### Verified

- The test suite in a fresh Python 3.13 environment on macOS (Apple silicon).
- Studio by hand in Chrome, every panel, on the synthetic room.
- Real footage end to end, with MapAnything and SAM 3 on Apple MPS: one phone
  video becomes a scene of 107 separate objects in about two and a half
  minutes, from the command line and from Studio. No ground truth, so this
  says nothing about accuracy.
  See [docs/BACKENDS.md](docs/BACKENDS.md#the-first-real-scene).

### Fixed while verifying

- Studio: labels in the 3D view could come up empty, or throw a DOM error,
  when the view was rebuilt. They now render into their own layer.
- Studio: the camera frames the objects instead of the whole room, so a scene
  opens at a useful zoom.
- Studio: a selected object keeps its colour instead of being tinted.
- Studio: **Duplicate** puts the copy next to the original, not inside it.
- Studio: uploading real footage without the model backends says which
  backend is missing.
- Command line: an unknown object or a missing scene gives a one-line error
  and exit code 2 instead of a traceback. Every subcommand has a help line.
- `Scene.summary()` returns plain Python numbers, not NumPy scalars.
- Geometry: confidence is mapped by rank within a frame. The earlier linear
  rescaling threw away most of a large plain surface in some frames.
- Physics: an object that ends a few centimetres below an uneven floor rests
  on the floor, not on the nearest wall.
- Studio: a crowded scene shows name tags for the 12 largest objects, plus
  the hovered and the selected one.
- Studio: the Scene tab says how many of the walls are square to the axes
  instead of claiming that all are.
- Studio: SAM 3 is reported as installed when `transformers` provides it.

### Known gaps

- The SAM 3D Objects wrapper has not been run on real weights, and nothing has
  been run on CUDA. See section 0 of [docs/DESIGN.md](docs/DESIGN.md).
- Real footage has been run on one clip, without ground truth.
- Parts and contents (shelf boards, books in a bookcase) are separate objects
  that overlap their container.
- Without shape completion, an object whose base was not seen can be reported
  as resting on a wall, or as floating.
- All acceptance thresholds and uncertainty constants were set on synthetic data.
