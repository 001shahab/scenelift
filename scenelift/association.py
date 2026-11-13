# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Step 6: decide which 2D masks in different views are the same physical object.

Each mask is lifted to 3D with the metric depth. Two masks belong together when
their 3D voxels overlap. If the segmenter already tracks objects across frames
(Mask2D.track_id), that identity is trusted first and geometry only merges
tracks that were split.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .geometry import project, unproject, voxel_keys
from .scale import usable_mask
from .types import Instance3D, Mask2D, View

_OFF = 1 << 20


def pack_voxels(points: np.ndarray, voxel: float) -> np.ndarray:
    """Sorted unique int64 key per occupied voxel."""
    k = voxel_keys(points, voxel) + _OFF
    return np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2])


def clean_masks(masks: list[Mask2D], cfg: Config) -> list[Mask2D]:
    """Drop tiny masks and duplicates of the same region (keeps the higher score)."""
    masks = [m for m in masks if m.mask.sum() >= cfg.mask_min_pixels]
    masks.sort(key=lambda m: -m.score)
    kept: list[Mask2D] = []
    for m in masks:
        area = m.mask.sum()
        dup = False
        for k in kept:
            inter = np.logical_and(m.mask, k.mask).sum()
            if inter / (area + k.mask.sum() - inter + 1e-9) > cfg.mask_nms_iou:
                dup = True
                break
        if not dup:
            kept.append(m)
    return kept


def lift_mask(view: View, mask: Mask2D, cfg: Config):
    """Mask -> (points, colors, conf). The mask is eroded so border pixels,
    where depth mixes object and background, never enter the object."""
    m = mask.mask.astype(np.uint8)
    if cfg.mask_erode_px > 0:
        k = 2 * cfg.mask_erode_px + 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
    m = m.astype(bool) & usable_mask(view, cfg)
    pts, v, u = unproject(view.depth, view.K, view.cam2world, m)
    return pts, view.image[v, u], view.conf[v, u]


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / min(len(a), len(b))


def _needed(a_len: int, b_len: int, same_label: bool, cfg: Config) -> float:
    """Overlap required to merge. Stricter when the two pieces differ a lot in
    size or carry different labels: that is the mug-on-table situation, where a
    small object shares its contact voxels with a big one."""
    need = cfg.assoc_overlap
    ratio = min(a_len, b_len) / max(a_len, b_len)
    if ratio < 0.2:
        need = max(need, 0.7)
    if not same_label:
        need = max(need, 0.5)
    return need


def _projection_agreement(a: Instance3D, b: Instance3D, views_by_index: dict, max_obs: int = 3) -> float:
    """Share of a's measured points that project inside b's masks.

    This is the wide-baseline test. Two photos taken from opposite sides see
    different faces of an object, so their 3D points do not overlap, but each
    set still projects inside the other's outline. Points far behind b's
    visible surface (more than b's own size) belong to something else.
    """
    pts = np.concatenate([c[0] for c in a.chunks])
    if len(pts) > 2000:
        pts = pts[:: len(pts) // 2000]
    b_pts = np.concatenate([c[0] for c in b.chunks])
    thickness = float(np.linalg.norm(np.ptp(b_pts, axis=0))) + 0.05
    fractions = []
    for m in sorted(b.observations, key=lambda o: -o.mask.sum())[:max_obs]:
        view = views_by_index[m.view_index]
        u, v, z = project(pts, view.K, view.world2cam)
        H, W = view.size
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if ok.mean() < 0.3:
            continue
        grown = cv2.dilate(m.mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        inside = grown[vi[ok], ui[ok]] & (z[ok] - view.depth[vi[ok], ui[ok]] < thickness)
        fractions.append(inside.mean())
    return float(np.mean(fractions)) if fractions else 0.0


def associate(views: list[View], masks_per_view: list[list[Mask2D]], cfg: Config) -> list[Instance3D]:
    instances: list[Instance3D] = []
    by_track: dict[int, Instance3D] = {}
    for view, masks in zip(views, masks_per_view):
        used = set()
        for m in clean_masks(masks, cfg):
            pts, cols, conf = lift_mask(view, m, cfg)
            if len(pts) < 20:
                continue
            vox = pack_voxels(pts, cfg.assoc_voxel)
            target = by_track.get(m.track_id) if m.track_id is not None else None
            if target is None:
                best, best_ov = None, 0.0
                for inst in instances:
                    if inst.id in used:
                        continue  # one object cannot own two masks in the same view
                    ov = _overlap(vox, inst.voxels)
                    if ov > best_ov and ov >= _needed(len(vox), len(inst.voxels),
                                                      inst.label == m.label, cfg):
                        best, best_ov = inst, ov
                target = best
            if target is None:
                target = Instance3D(id=len(instances), voxels=vox, track_id=m.track_id)
                instances.append(target)
            else:
                target.voxels = np.union1d(target.voxels, vox)
            if m.track_id is not None:
                by_track[m.track_id] = target
            used.add(target.id)
            target.observations.append(m)
            target.chunks.append((pts, cols, conf))
            target.label_votes[m.label] = target.label_votes.get(m.label, 0.0) + m.score * m.mask.sum()

    # final pass: merge tracks that turned out to be the same object
    views_by_index = {v.index: v for v in views}
    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            a, b = instances[i], instances[j]
            shared_view = {o.view_index for o in a.observations} & {o.view_index for o in b.observations}
            if shared_view:
                continue  # seen as two separate masks in one image: keep them apart
            if _overlap(a.voxels, b.voxels) >= _needed(len(a.voxels), len(b.voxels),
                                                       a.label == b.label, cfg):
                parent[find(j)] = find(i)
            elif a.label == b.label and min(_projection_agreement(a, b, views_by_index),
                                            _projection_agreement(b, a, views_by_index)) >= cfg.assoc_projection:
                parent[find(j)] = find(i)
    merged: dict[int, Instance3D] = {}
    for i, inst in enumerate(instances):
        r = find(i)
        if r not in merged:
            merged[r] = Instance3D(id=len(merged), voxels=inst.voxels, track_id=inst.track_id)
        tgt = merged[r]
        tgt.voxels = np.union1d(tgt.voxels, inst.voxels)
        tgt.observations += inst.observations
        tgt.chunks += inst.chunks
        for k, v in inst.label_votes.items():
            tgt.label_votes[k] = tgt.label_votes.get(k, 0.0) + v

    min_views = cfg.min_views_per_object if len(views) >= 3 else 1
    return [inst for inst in merged.values() if inst.n_views >= min_views]
