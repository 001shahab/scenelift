# Contributing to scenelift

Thank you for considering a contribution. scenelift is developed by
Prof. Shahab Anbarjafari at 3S Holding OÜ and is open to improvements from
everyone: bug reports, results on real footage, new backends, documentation.

## The most useful contributions right now

1. **Results on real scenes.** Run the pipeline on your own footage, measure a
   few lengths with a tape, and report `scenelift eval` output. This is
   milestone M0 of the [roadmap](docs/DESIGN.md#12-roadmap-to-a-release-people-will-star).
2. **Fixes to the backend wrappers** in `scenelift/backends/`. They follow the
   upstream READMEs but have not met real weights yet.
3. **Permissively licensed backends** behind the same three interfaces
   (`scenelift/backends/base.py`): an Apache or MIT segmenter, an MIT
   image-to-3D model.
4. **Failure cases.** A scene that breaks a step, with the frames attached, is
   worth more than a feature.

## Set up

```bash
git clone https://github.com/001shahab/scenelift.git
cd scenelift
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

There is one requirements file. Please do not add another.

The front end lives in `frontend/` and needs Node.js 20 or newer:

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, talks to `python app.py` on port 8000
npm run build      # typecheck, then write the bundle to scenelift/studio/static
```

Commit the rebuilt bundle together with the source change, so that
`python app.py` keeps working for people without Node.js.

## Ground rules for code

These come straight from the [design principles](docs/DESIGN.md#3-design-principles).

- **Measure first, generate second.** Pose and scale come from multi-view
  measurement. A generative model may only fill in what no camera saw.
- **Never hide a guess.** Anything that returns a length returns a sigma with
  it, and says when it relies on generated geometry. The user interface never
  shows a number without its error bar.
- **Every tunable number lives in `scenelift/config.py`**, with a comment that
  says what it does and in which unit.
- **The core must run on a laptop CPU.** No import of `torch` outside
  `scenelift/backends/`.
- **Each step can be tested alone.** A change to a step comes with a test on
  the synthetic room, which has exact ground truth.
- Units are metres, the world is Z-up with the floor at z = 0, cameras are
  OpenCV (`+X` right, `+Y` down, `+Z` forward), pixel `(u, v)` is
  `(column, row)`.
- The Studio server (`scenelift/studio/`) stays a thin wrapper. If a route
  needs logic, the logic belongs in the library, with a test there.

Style: plain, readable Python with type hints on public functions, lines up to
110 characters, comments that explain why. New files start with:

```python
# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0
```

## Pull requests

1. Open an issue first for anything larger than a small fix, so the approach
   can be agreed before you spend time on it.
2. Keep a pull request to one topic.
3. Run `pytest -q`, and `npm run build` if you touched `frontend/`.
4. Describe what changed, why, and how you tested it. Say plainly what you did
   not test.
5. Add a line to `CHANGELOG.md` under an "Unreleased" heading.

## Licence of contributions

scenelift is licensed under the Apache License 2.0, copyright 3S Holding OÜ.
By submitting a contribution you agree that it is licensed under the same
terms (section 5 of the licence), and you confirm that you have the right to
submit it.

## Questions

Open a [discussion or an issue](https://github.com/001shahab/scenelift/issues),
or write to shb@3sholding.com.
