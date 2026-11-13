# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Benchmark helpers: is the scale right, and are the error bars honest?

Ground truth is a JSON list made with a tape measure, for example:
[
  {"type": "dimension", "object": "table_1", "axis": "length", "true_m": 1.60},
  {"type": "distance", "a": "sofa_1", "b": "table_1", "mode": "surface", "true_m": 0.45},
  {"type": "fit", "object": "chair_1", "space": [0.6, 0.6, 1.0], "true": true}
]
"""

from __future__ import annotations

import json

import numpy as np

from . import query


def evaluate(scene, ground_truth) -> dict:
    """Returns accuracy and calibration numbers.

    abs_rel_error:   mean |measured - true| / true            (accuracy)
    coverage_68/95:  share of truths inside the 1 and 1.96 sigma intervals.
                     Honest error bars give about 0.68 and 0.95. Lower means
                     overconfident, higher means the bars are too wide.
    mean_z:          mean signed (measured - true) / sigma; far from 0 means a
                     shared scale bias.
    fit_accuracy / fit_brier: for yes/no fit questions.
    """
    if isinstance(ground_truth, str):
        with open(ground_truth) as f:
            ground_truth = json.load(f)
    rel, z, rows = [], [], []
    fit_p, fit_y = [], []
    for g in ground_truth:
        if g["type"] == "dimension":
            m = query.dimensions(scene, g["object"])[g["axis"]]
        elif g["type"] == "distance":
            m = query.distance(scene, g["a"], g["b"], g.get("mode", "surface"))
        elif g["type"] == "fit":
            r = query.fits_in_space(scene, g["object"], g["space"])
            fit_p.append(r.probability)
            fit_y.append(1.0 if g["true"] else 0.0)
            rows.append({**g, "probability": r.probability, "verdict": r.verdict})
            continue
        else:
            raise ValueError(f"unknown ground-truth type: {g['type']}")
        t = float(g["true_m"])
        rel.append(abs(m.value - t) / t)
        z.append((m.value - t) / max(m.sigma, 1e-9))
        rows.append({**g, "measured_m": m.value, "sigma_m": m.sigma})
    out = {"n_lengths": len(rel), "n_fits": len(fit_p), "rows": rows}
    if rel:
        za = np.abs(z)
        out.update(abs_rel_error=float(np.mean(rel)), median_rel_error=float(np.median(rel)),
                   coverage_68=float(np.mean(za <= 1.0)), coverage_95=float(np.mean(za <= 1.96)),
                   mean_z=float(np.mean(z)))
    if fit_p:
        p, y = np.array(fit_p), np.array(fit_y)
        out.update(fit_accuracy=float(np.mean((p >= 0.5) == (y > 0.5))),
                   fit_brier=float(np.mean((p - y) ** 2)))
    return out
