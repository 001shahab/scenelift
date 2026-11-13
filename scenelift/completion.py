# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 8: give every object a complete shape without lying about what was seen.

A generative backend proposes a full mesh from one image. scenelift then
  1. registers that mesh to the measured points (pose and scale come from the
     measurement, never from the generator),
  2. checks it against free space: no part of the mesh may sit where a camera
     saw straight through,
  3. tags each vertex as observed or generated.
If the proposal fails the checks, the object falls back to the convex hull of
its measured points and is flagged.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import Config
from .geometry import icp_partial, make_transform, project, rot_z, transform_points
from .types import CompletedShape, Instance3D, Mask2D, View


def cube_rotations() -> list:
    """The 24 rotations that map the coordinate axes onto each other."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def best_observation(inst: Instance3D, views: list[View]) -> tuple[View, Mask2D]:
    """The view where the object is largest, most confident and not cut by the image border."""
    by_index = {v.index: v for v in views}
    best, best_score = None, -1.0
    for m in inst.observations:
        v = by_index[m.view_index]
        area = float(m.mask.sum())
        conf = float(v.conf[m.mask].mean()) if area else 0.0
        border = m.mask[0].any() or m.mask[-1].any() or m.mask[:, 0].any() or m.mask[:, -1].any()
        score = area * conf * (0.5 if border else 1.0)
        if score > best_score:
            best, best_score = (v, m), score
    return best


def freespace_violation(points: np.ndarray, views: list[View], cfg: Config, max_views: int = 8) -> float:
    """Fraction of points that sit in front of a measured surface.

    If a camera measured a surface 3 m away along some pixel, the space before
    it is empty. A proposed mesh that puts material there contradicts the data.
    """
    if not views:
        return 0.0
    sel = views if len(views) <= max_views else [views[i] for i in
                                                 np.linspace(0, len(views) - 1, max_views).round().astype(int)]
    bad = total = 0
    for v in sel:
        u, vv, z = project(points, v.K, v.world2cam)
        H, W = v.size
        ui, vi = np.round(u).astype(int), np.round(vv).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not ok.any():
            continue
        d = v.depth[vi[ok], ui[ok]]
        valid = v.valid[vi[ok], ui[ok]] & (d > 0)
        tol = np.maximum(cfg.freespace_tol, 0.03 * d)
        bad += int(np.count_nonzero(valid & (z[ok] < d - tol)))
        total += int(np.count_nonzero(valid))
    return bad / total if total else 0.0


def register(shape: CompletedShape, observed: np.ndarray, views: list[View], cfg: Config,
             rng: np.random.Generator, init: Optional[np.ndarray] = None):
    """Find the similarity that places the generated mesh on the measured points.

    Tries 48 starting orientations (24 axis-aligned ones, each with and without
    a 45 degree turn about the vertical), refines the best few with partial
    ICP, and ranks them by fit plus free-space violation.
    Returns (T, residual, violation).
    """
    mesh = trimesh.Trimesh(shape.vertices, shape.faces, process=False)
    model, _ = trimesh.sample.sample_surface(mesh, cfg.icp_points, seed=int(rng.integers(1 << 30)))
    model = np.asarray(model)
    obs = observed if len(observed) <= cfg.icp_points else observed[
        rng.choice(len(observed), cfg.icp_points, replace=False)]
    obs_small = obs[:: max(1, len(obs) // 500)]
    model_small = model[:: max(1, len(model) // 800)]
    diag = float(np.linalg.norm(np.ptp(obs, axis=0))) + 1e-9
    o_lo, o_hi = np.percentile(obs, 1, axis=0), np.percentile(obs, 99, axis=0)

    inits = []
    if init is not None:
        inits.append(init)
    for R0 in cube_rotations():
        for yaw in (0.0, np.pi / 4):
            R = rot_z(yaw) @ R0
            mr = model_small @ R.T
            m_lo, m_hi = mr.min(axis=0), mr.max(axis=0)
            s = float(np.linalg.norm(o_hi - o_lo) / (np.linalg.norm(m_hi - m_lo) + 1e-9))
            t = 0.5 * (o_lo + o_hi) - s * 0.5 * (m_lo + m_hi)
            inits.append(make_transform(R, t, s))

    coarse = []
    for T0 in inits:
        T, res = icp_partial(model_small, obs_small, T0, iters=8, trim=cfg.icp_trim)
        coarse.append((res, T))
    coarse.sort(key=lambda c: c[0])

    best = None
    for _, T0 in coarse[:5]:
        T, res = icp_partial(model, obs, T0, iters=cfg.icp_iters, trim=cfg.icp_trim)
        viol = freespace_violation(transform_points(T, model), views, cfg)
        score = res / diag + viol
        if best is None or score < best[0]:
            best = (score, T, res, viol)
    _, T, res, viol = best
    return T, res, viol


def silhouette_iou(points: np.ndarray, observations: list, cfg: Config) -> float:
    """Mean overlap between the projected shape and the 2D masks.

    observations: list of (View, Mask2D). Only the part of the shape that the
    camera could see (not hidden behind a nearer measured surface) is drawn, so
    an occluding object does not count against the shape.
    """
    import cv2

    ious = []
    for view, m in observations:
        u, v, z = project(points, view.K, view.world2cam)
        H, W = view.size
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, z = ui[ok], vi[ok], z[ok]
        d = view.depth[vi, ui]
        vis = ~view.valid[vi, ui] | (z <= d + np.maximum(cfg.freespace_tol, 0.03 * d))
        sil = np.zeros((H, W), np.uint8)
        sil[vi[vis], ui[vis]] = 1
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        union = np.logical_or(sil, m.mask).sum()
        if union:
            ious.append(np.logical_and(sil, m.mask).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def refine_scale(T: np.ndarray, shape: CompletedShape, observed: np.ndarray, observations: list,
                 cfg: Config, rng: np.random.Generator):
    """Fix the size with the image outlines. Returns (T, outline_iou, residual).

    Measured points come from the inside of each mask, so they stop a pixel or
    two short of the true edge, and a shape fitted to them alone comes out a few
    percent small. The outline of the object in the images does not have that
    bias. This searches the scale that best reproduces the outlines, re-fitting
    position and rotation to the measured points at every step.
    """
    mesh = trimesh.Trimesh(shape.vertices, shape.faces, process=False)
    if len(observations) > cfg.silhouette_views:
        observations = sorted(observations, key=lambda o: -o[1].mask.sum())[: cfg.silhouette_views]
    # enough samples that the projected shape is solid: several points per mask pixel
    n_dense = int(np.clip(12 * max(o[1].mask.sum() for o in observations), 20000, 250000))
    dense = np.asarray(trimesh.sample.sample_surface(mesh, n_dense, seed=int(rng.integers(1 << 30)))[0])
    model = dense[:: max(1, len(dense) // cfg.icp_points)]
    obs = observed if len(observed) <= cfg.icp_points else observed[
        rng.choice(len(observed), cfg.icp_points, replace=False)]
    best_T, best_iou = T, silhouette_iou(transform_points(T, dense), observations, cfg)
    c = transform_points(T, dense).mean(axis=0)
    lo, hi, n = cfg.silhouette_scales
    for s in np.linspace(lo, hi, int(n)):
        Ts = make_transform(t=c) @ make_transform(s=float(s)) @ make_transform(t=-c) @ T
        Ts, _ = icp_partial(model, obs, Ts, iters=10, trim=cfg.icp_trim, with_scale=False)
        iou = silhouette_iou(transform_points(Ts, dense), observations, cfg)
        if iou > best_iou + 0.003:
            best_T, best_iou = Ts, iou
    residual = float(np.median(cKDTree(transform_points(best_T, dense)).query(obs)[0]))
    return best_T, best_iou, residual


def tag_provenance(vertices: np.ndarray, observed: np.ndarray, observed_colors: np.ndarray,
                   base_colors: Optional[np.ndarray], cfg: Config):
    """Mark each vertex observed or generated, and paint observed vertices with
    the real measured colour."""
    dist, nn = cKDTree(observed).query(vertices)
    is_obs = dist < cfg.observed_tau
    colors = (np.full((len(vertices), 3), 160, np.uint8) if base_colors is None
              else base_colors[:, :3].astype(np.uint8).copy())
    colors[is_obs] = observed_colors[nn[is_obs]]
    return is_obs, colors


def hull_mesh(points: np.ndarray):
    """Convex hull of the measured points: the honest minimum when nothing better exists."""
    try:
        h = trimesh.Trimesh(points, process=False).convex_hull
        return np.asarray(h.vertices), np.asarray(h.faces)
    except Exception:
        lo, hi = points.min(axis=0), points.max(axis=0)
        b = trimesh.creation.box(extents=np.maximum(hi - lo, 1e-3))
        b.apply_translation(0.5 * (lo + hi))
        return np.asarray(b.vertices), np.asarray(b.faces)


def complete_object(obj, inst: Instance3D, views: list[View], completer, cfg: Config,
                    rng: np.random.Generator) -> None:
    """Fill obj.mesh_* in place. obj is a scene.SceneObject."""
    shape = None
    view = mask = None
    if completer is not None:
        view, mask = best_observation(inst, views)
        try:
            shape = completer.complete(view, mask)
        except Exception as e:  # a failing backend must not kill the whole scene
            obj.flags.append(f"completion_error:{type(e).__name__}")
    if shape is not None and len(shape.faces) > 0:
        init = view.cam2world @ shape.pose_cam if shape.pose_cam is not None else None
        T, res, viol = register(shape, obj.points, views, cfg, rng, init)
        diag = float(np.linalg.norm(np.ptp(obj.points, axis=0)))
        by_index = {v.index: v for v in views}
        T, iou, res = refine_scale(T, shape, obj.points,
                              [(by_index[m.view_index], m) for m in inst.observations], cfg, rng)
        verts = transform_points(T, shape.vertices)
        viol = freespace_violation(verts, views, cfg)
        obj.fit = {"residual_m": round(res, 4), "freespace_violation": round(viol, 4),
                   "outline_iou": round(iou, 4)}
        if (viol <= cfg.max_freespace_violation and iou - viol >= cfg.min_fit_quality
                and res <= max(cfg.max_fit_residual, cfg.max_fit_residual_rel * diag)):
            is_obs, cols = tag_provenance(verts, obj.points, obj.colors, shape.colors, cfg)
            obj.mesh_vertices, obj.mesh_faces = verts, np.asarray(shape.faces)
            obj.mesh_colors, obj.vertex_observed = cols, is_obs
            obj.completion = "generated"
            obj.update_obb()
            return
        obj.flags.append("completion_rejected")
    verts, faces = hull_mesh(obj.points)
    is_obs, cols = tag_provenance(verts, obj.points, obj.colors, None, cfg)
    obj.mesh_vertices, obj.mesh_faces = verts, faces
    obj.mesh_colors, obj.vertex_observed = cols, is_obs
    obj.completion = "hull"
    obj.update_obb()
