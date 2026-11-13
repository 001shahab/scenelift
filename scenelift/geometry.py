# Copyright 2026 3S Holding OÜ
# SPDX-License-Identifier: Apache-2.0

"""Pure numpy/scipy geometry used by every stage. No learned models here."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------
def unproject(depth: np.ndarray, K: np.ndarray, cam2world: np.ndarray,
              mask: Optional[np.ndarray] = None):
    """Lift pixels to world points. Returns (points (N,3), v (N,), u (N,))."""
    if mask is None:
        mask = np.ones(depth.shape, dtype=bool)
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    return transform_points(cam2world, cam), v, u


def project(points: np.ndarray, K: np.ndarray, world2cam: np.ndarray):
    """Project world points. Returns (u, v) as floats and z in the camera frame."""
    cam = transform_points(world2cam, points)
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return u, v, z


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def make_transform(R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None,
                   s: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = s * np.asarray(R)
    else:
        T[:3, :3] *= s
    if t is not None:
        T[:3, 3] = t
    return T


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def depth_edge_mask(depth: np.ndarray, rel: float) -> np.ndarray:
    """True where depth is smooth. Removes the 'flying pixels' at object borders."""
    d = depth.astype(np.float32)
    dmax = cv2.dilate(d, np.ones((3, 3), np.uint8))
    dmin = cv2.erode(d, np.ones((3, 3), np.uint8))
    return (dmax - dmin) <= rel * np.maximum(d, 1e-6)


# --------------------------------------------------------------------------
# point clouds
# --------------------------------------------------------------------------
def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(points / voxel).astype(np.int64)


def voxel_downsample(points: np.ndarray, voxel: float, colors: Optional[np.ndarray] = None,
                     weights: Optional[np.ndarray] = None):
    """Average all points falling in each voxel. Returns (points, colors, counts)."""
    if len(points) == 0:
        return points, colors, np.zeros(0, dtype=np.int64)
    keys = voxel_keys(points, voxel)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    w = np.ones(len(points)) if weights is None else np.maximum(weights, 1e-6)
    wsum = np.bincount(inv, weights=w)
    out = np.stack([np.bincount(inv, weights=points[:, i] * w) / wsum for i in range(3)], axis=1)
    out_c = None
    if colors is not None:
        out_c = np.stack([np.bincount(inv, weights=colors[:, i] * w) / wsum for i in range(3)],
                         axis=1).astype(np.uint8)
    return out, out_c, counts


def statistical_outlier_mask(points: np.ndarray, k: int = 12, std: float = 2.0) -> np.ndarray:
    """True for points whose mean k-neighbour distance is not unusually large."""
    n = len(points)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    d, _ = cKDTree(points).query(points, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    med = np.median(mean_d)
    mad = 1.4826 * np.median(np.abs(mean_d - med))
    # on a clean, evenly sampled surface the spread is almost zero; the floor on
    # it keeps the test from eating the edges of the object
    return mean_d <= med + std * max(mad, 0.25 * med)


def largest_cluster_mask(points: np.ndarray, radius: float) -> np.ndarray:
    """Keep the biggest connected component (points linked within `radius`)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(points)
    if n < 3:
        return np.ones(n, dtype=bool)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(n, dtype=bool)
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------
def fit_plane_svd(points: np.ndarray):
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[2]
    return n, -float(n @ c)


def ransac_plane(points: np.ndarray, thresh: float, iters: int, rng: np.random.Generator,
                 normal_prior: Optional[np.ndarray] = None, max_angle_deg: float = 180.0,
                 perpendicular: bool = False, below_penalty: float = 0.0,
                 max_eval: int = 20000):
    """RANSAC plane n.x + d = 0.

    normal_prior + max_angle_deg restrict the normal to a cone around the prior;
    with perpendicular=True the normal must instead be within max_angle_deg of
    being perpendicular to the prior (used for walls).
    below_penalty > 0 penalises points on the negative side of the plane, which
    is what makes the floor win over table tops and the ceiling.
    Returns (n, d, inlier_mask) or None.
    """
    n_pts = len(points)
    if n_pts < 50:
        return None
    sub = points if n_pts <= max_eval else points[rng.choice(n_pts, max_eval, replace=False)]
    idx = rng.integers(0, len(sub), size=(iters, 3))
    p0, p1, p2 = sub[idx[:, 0]], sub[idx[:, 1]], sub[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-9
    normals[ok] /= norm[ok, None]
    if normal_prior is not None:
        prior = normal_prior / np.linalg.norm(normal_prior)
        cosang = normals @ prior
        if perpendicular:
            ok &= np.abs(cosang) <= np.sin(np.deg2rad(max_angle_deg))
        else:
            flip = cosang < 0
            normals[flip] *= -1
            ok &= np.abs(cosang) >= np.cos(np.deg2rad(max_angle_deg))
    best, best_score = None, -np.inf
    for i in np.nonzero(ok)[0]:
        n = normals[i]
        d = -float(n @ p0[i])
        dist = sub @ n + d
        score = float(np.count_nonzero(np.abs(dist) < thresh))
        if below_penalty > 0:
            score -= below_penalty * float(np.count_nonzero(dist < -3 * thresh))
        if score > best_score:
            best_score, best = score, (n, d)
    if best is None:
        return None
    n, d = best
    inl = np.abs(points @ n + d) < thresh
    if inl.sum() >= 10:
        n2, d2 = fit_plane_svd(points[inl])
        if n2 @ n < 0:
            n2, d2 = -n2, -d2
        n, d = n2, d2
        inl = np.abs(points @ n + d) < thresh
    return n, d, inl


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Least-squares similarity T with dst ~= T(src)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = 1.0
    if with_scale:
        var = (xs ** 2).sum() / len(src)
        s = float(np.trace(np.diag(S) @ D) / max(var, 1e-12))
    t = mu_d - s * R @ mu_s
    return make_transform(R, t, s)


def similarity_scale(T: np.ndarray) -> float:
    return float(np.cbrt(abs(np.linalg.det(T[:3, :3]))))


def icp_partial(model: np.ndarray, observed: np.ndarray, T_init: np.ndarray, iters: int = 30,
                trim: float = 0.85, with_scale: bool = True,
                scale_limits: tuple = (0.5, 2.0)):
    """Register a complete model to a partial observation.

    Finds a similarity T such that every observed point lies on T(model). The
    matching runs one way only (observed -> model), so the unseen back of the
    model is never asked to match anything. Returns (T, median_residual).
    """
    T = T_init.copy()
    s0 = similarity_scale(T_init)
    res = np.inf
    for _ in range(iters):
        mw = transform_points(T, model)
        dist, nn = cKDTree(mw).query(observed)
        keep = dist <= np.quantile(dist, trim)
        if keep.sum() < 6:
            break
        step = umeyama(mw[nn[keep]], observed[keep], with_scale=with_scale)
        cand = step @ T
        ratio = similarity_scale(cand) / s0
        if not (scale_limits[0] <= ratio <= scale_limits[1]):
            step = umeyama(mw[nn[keep]], observed[keep], with_scale=False)
            cand = step @ T
        new_res = float(np.median(dist))
        T = cand
        if abs(res - new_res) < 1e-6:
            res = new_res
            break
        res = new_res
    mw = transform_points(T, model)
    dist, _ = cKDTree(mw).query(observed)
    return T, float(np.median(dist))


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------
def gravity_obb(points: np.ndarray) -> dict:
    """Tightest box that stays upright (rotation about Z only).

    Returns dict(center (3,), size (3,) = (length, width, height), yaw).
    """
    xy = points[:, :2].astype(np.float32)
    if len(xy) >= 3:
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy)
    else:
        cx, cy = xy.mean(axis=0)
        w = h = 1e-3
        ang = 0.0
    yaw = float(np.deg2rad(ang))
    if h > w:  # keep the first size as the longer horizontal side
        w, h = h, w
        yaw += np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    z0, z1 = float(points[:, 2].min()), float(points[:, 2].max())
    return {
        "center": np.array([cx, cy, 0.5 * (z0 + z1)], dtype=np.float64),
        "size": np.array([max(w, 1e-3), max(h, 1e-3), max(z1 - z0, 1e-3)], dtype=np.float64),
        "yaw": yaw,
    }


def obb_footprint(obb: dict) -> np.ndarray:
    """(4, 2) corners of the box footprint on the floor, counter-clockwise."""
    c, s = np.cos(obb["yaw"]), np.sin(obb["yaw"])
    ax = np.array([c, s]) * obb["size"][0] / 2
    ay = np.array([-s, c]) * obb["size"][1] / 2
    ctr = obb["center"][:2]
    return np.stack([ctr - ax - ay, ctr + ax - ay, ctr + ax + ay, ctr - ax + ay])


def obb_corners(obb: dict) -> np.ndarray:
    fp = obb_footprint(obb)
    z0 = obb["center"][2] - obb["size"][2] / 2
    z1 = obb["center"][2] + obb["size"][2] / 2
    return np.concatenate([np.c_[fp, np.full(4, z0)], np.c_[fp, np.full(4, z1)]])


def points_in_footprint(pts_xy: np.ndarray, obb: dict, margin: float = 0.0) -> np.ndarray:
    c, s = np.cos(obb["yaw"]), np.sin(obb["yaw"])
    rel = pts_xy - obb["center"][:2]
    a = rel @ np.array([c, s])
    b = rel @ np.array([-s, c])
    return (np.abs(a) <= obb["size"][0] / 2 + margin) & (np.abs(b) <= obb["size"][1] / 2 + margin)


def footprint_mtv(fp_a: np.ndarray, fp_b: np.ndarray):
    """Separating-axis test for two convex quads.

    Returns None when they do not overlap, otherwise the minimum translation
    vector (2,) that moves A out of B.
    """
    best_overlap, best_axis = np.inf, None
    for quad in (fp_a, fp_b):
        for i in range(4):
            e = quad[(i + 1) % 4] - quad[i]
            axis = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(axis)
            if nrm < 1e-12:
                continue
            axis /= nrm
            pa, pb = fp_a @ axis, fp_b @ axis
            overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
            if overlap <= 0:
                return None
            if overlap < best_overlap:
                best_overlap = overlap
                best_axis = axis if pa.mean() >= pb.mean() else -axis
    return best_axis * best_overlap
