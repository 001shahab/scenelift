# Model backends

scenelift's own code runs on a laptop CPU and needs no model weights. The
synthetic room in SceneLift Studio and the whole test suite work with
`pip install -r requirements.txt` alone.

Reconstructing **real footage** needs three kinds of model. Each sits behind a
small interface in [`scenelift/backends/base.py`](../scenelift/backends/base.py),
is installed separately from its own repository, and is never bundled with
scenelift.

| Role | Interface | Default | Needed |
|---|---|---|---|
| Geometry: metric depth, intrinsics and pose per frame | `GeometryBackend.infer(frames) -> list[View]` | MapAnything | yes |
| Segmentation: 2D instance masks per frame | `SegmentationBackend.segment(views) -> list[list[Mask2D]]` | SAM 3 | yes |
| Completion: a full mesh from one image and one mask | `CompletionBackend.complete(view, mask) -> CompletedShape or None` | SAM 3D Objects | no |

Without a completion backend every object gets the convex hull of its measured
points, clearly marked as such. Nothing is invented.

## Status: read this first

The three wrappers follow the usage shown in each project's README and demo
code, read on 13 November 2026.

| Wrapper | State |
|---|---|
| `backends/mapanything.py` | **Run on real weights**, unchanged, on an Apple-silicon laptop (MPS). One clip so far |
| `backends/sam3.py` | **Not run yet.** The checkpoint is gated on Hugging Face and access is granted by hand. Expect small fixes on first contact |
| `backends/sam3d.py` | **Not run yet.** Gated checkpoint, and the code requires CUDA. Expect small fixes on first contact |

So today real footage gives metric geometry, cameras, the floor and the walls.
Separate objects need SAM 3 (or another segmenter behind the same interface),
and that step has not been seen working on real images. A pull request with
fixes, or with a result, is very welcome.

### What was observed with MapAnything

One run, 13 November 2026. MapAnything 1.1.4 (commit `3d10cf7`), PyTorch 2.15,
Python 3.13, a MacBook with an M5 Max and 36 GB of memory, device `mps`. The
input was a 12.8 second hand-held phone video of a furnished room, 3840 x 2160
HEVC at 60 frames per second, filmed in portrait.

| Step | Result |
|---|---|
| Keyframe selection (`ingest.load`) | 24 sharp keyframes, rotated upright, 720 x 1280, in about 10 s |
| Loading the model | 106 s the first time, most of it the 4.9 GB checkpoint download |
| Inference | 11.4 s for 24 frames, depth maps of 294 x 518 |
| Depth | 88 to 99 percent of the pixels valid per frame, medians between 1.3 and 3.2 m, which is plausible for the room |
| Scale harmonisation | The 24 views agree on scale to within 0.96 percent after correction |
| Layout | Floor found, 4 wall planes, about 109,000 background points, 1.5 s on the CPU |
| Studio | The saved scene opens: point cloud, 24 cameras, 24 keyframes |

What this does **not** show: accuracy. Nothing in that room was measured with
a tape, so the absolute scale is unchecked.

Two things to look at next, both seen in this run and not yet investigated:

1. **The focal length drifts.** A phone lens is fixed, yet the estimated focal
   length went from 473 px in sharp frames down to 252 px in the middle of a
   fast pan. Depth and focal length trade off against each other, so this
   deserves a shared-intrinsics constraint across the frames of one video.
2. **A wall was found twice**, as two nearly parallel planes a short distance
   apart. The wall thresholds were set on the synthetic room.

## Licences

| Backend | Code and weights | Note |
|---|---|---|
| MapAnything, checkpoint `facebook/map-anything-apache` | Apache 2.0 code, Apache 2.0 checkpoint | The default here |
| MapAnything, checkpoint `facebook/map-anything` | Not the Apache checkpoint | Read its model card before use |
| SAM 3 | SAM License | A custom licence, not an OSI open-source licence |
| SAM 3D Objects | SAM License | A custom licence, not an OSI open-source licence |

The SAM License has use restrictions and trade-control clauses. Read it in
full, ideally with legal advice, before depending on it for anything
commercial. scenelift's Apache 2.0 licence covers scenelift only. It gives you
no rights to the backends and their weights.

A default stack made of permissively licensed parts only is on the roadmap:
an Apache or MIT segmenter and an MIT image-to-3D model behind the same
interfaces, with the SAM backends as an opt-in.

## Hardware

| Mode | Geometry | Segmentation | Completion |
|---|---|---|---|
| Full | CUDA GPU | CUDA GPU | CUDA GPU (required by SAM 3D Objects) |
| Laptop | Apple MPS: works, see above. CPU: not tried | MPS or CPU: not tried | none: convex hulls, flagged |

## Install

Use the same virtual environment as scenelift. The commands below are the
ones given by each project at the time of writing. If they differ from the
project's current README, the README wins.

### 1. PyTorch

Install the build that matches your GPU from <https://pytorch.org/get-started/locally/>.

### 2. MapAnything (geometry)

```bash
git clone https://github.com/facebookresearch/map-anything
pip install -e map-anything
```

or, without a checkout:

```bash
pip install git+https://github.com/facebookresearch/map-anything.git
```

The checkpoint `facebook/map-anything-apache` (4.9 GB, public, no login) is
downloaded from Hugging Face on first use, into `~/.cache/huggingface`.

MapAnything pins some of its dependencies. In our environment its install
replaced `opencv-python-headless` 5.0 with 4.10. scenelift needs 4.8 or newer
and its test suite passes on both.

### 3. SAM 3 (segmentation)

```bash
git clone https://github.com/facebookresearch/sam3
pip install -e sam3
```

At the time of writing the SAM 3 checkpoints are gated on Hugging Face, with
manual approval: request access on the [model page](https://huggingface.co/facebook/sam3),
wait for it to be granted, and log in with `hf auth login` (or
`huggingface-cli login`) before the first run. Follow the SAM 3 README if this
has changed.

### 4. SAM 3D Objects (completion, optional)

Follow <https://github.com/facebookresearch/sam-3d-objects>. It needs a CUDA
GPU. Download the checkpoints into `<repo>/checkpoints/hf`, so that
`<repo>/checkpoints/hf/pipeline.yaml` exists. scenelift is then pointed at the
checkout:

```bash
python app.py --sam3d-repo /path/to/sam-3d-objects
scenelift run room.mp4 -o room --sam3d-repo /path/to/sam-3d-objects
```

or with the environment variable `SAM3D_REPO`.

## Check the installation

```bash
python app.py
```

The badge in the top bar reads "Model backends ready" once MapAnything and
SAM 3 can be imported, and `GET /api/system` lists each backend. Until both
are there, Studio refuses uploads of real footage and says which one is
missing. Then run the smoke test on a short video or a folder of photos:

```bash
SCENELIFT_TEST_INPUT=/path/to/room.mp4 pytest -q tests/test_backends_live.py
SCENELIFT_TEST_INPUT=/path/to/room.mp4 SAM3D_REPO=/path/to/sam-3d-objects \
    pytest -q tests/test_backends_live.py
```

| Test | Needs | Checks |
|---|---|---|
| `test_geometry_backend_contract` | MapAnything | Shapes and types, depth in metres, confidence in 0 to 1, intrinsics, a rigid pose |
| `test_geometry_through_the_core` | MapAnything | Real depth through scale harmonisation, gravity, floor and walls, fusion, save and load, with no masks |
| `test_real_pipeline` | MapAnything and SAM 3 | The whole pipeline, with objects. Skipped while SAM 3 is missing |

The first two run on the first 8 keyframes. Together they took 26 s on the
laptop above, with the checkpoint already cached.

## Write your own backend

A backend is any object with the right method. No base class, no registration.

```python
import numpy as np
from scenelift import Config, SceneLift
from scenelift.types import Frame, Mask2D, View


class MyDepthModel:
    def infer(self, frames: list[Frame]) -> list[View]:
        views = []
        for f in frames:
            depth, K, cam2world, conf = my_model(f.image)   # metres, OpenCV camera
            views.append(View(index=f.index, image=f.image, depth=depth.astype(np.float32),
                              K=K, cam2world=cam2world, conf=conf,
                              valid=np.isfinite(depth) & (depth > 0), source=f.source))
        return views


class MySegmenter:
    def segment(self, views: list[View]) -> list[list[Mask2D]]:
        return [[Mask2D(view_index=v.index, mask=m, label=name, score=s)
                 for m, name, s in my_masks(v.image)] for v in views]


scene = SceneLift(MyDepthModel(), MySegmenter(), completer=None, config=Config()).run("room.mp4")
```

What a geometry backend must return per frame:

- `depth`: z-depth in metres, at the resolution of `image`.
- `K`: 3 x 3 intrinsics for that resolution.
- `cam2world`: 4 x 4, OpenCV camera convention.
- `conf`: confidence in the range 0 to 1 (squash it per view, for example
  between the 2nd and 98th percentile, so one threshold works across scenes).
- `valid`: a boolean image, `False` wherever depth is non-finite or not positive.
- `image`: the frame at the same resolution as `depth`.

A segmenter that follows objects through a video should set `Mask2D.track_id`;
association then trusts that identity first. A completion backend may return
its mesh in any frame and at any scale: both are discarded, and the mesh is
registered to the measured points.

`scenelift/backends/synthetic.py` is a complete, small example of all three.
