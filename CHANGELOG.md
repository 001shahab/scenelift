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

### Known gaps

- The three model backend wrappers have not been run on real weights yet. See
  section 0 of [docs/DESIGN.md](docs/DESIGN.md).
- All acceptance thresholds and uncertainty constants were set on synthetic data.
