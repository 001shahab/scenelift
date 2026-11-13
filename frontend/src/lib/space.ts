// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Small geometry helpers in scenelift's conventions: metres, Z up, floor at
// z = 0, OpenCV cameras (+X right, +Y down, +Z forward).

import type { CameraDoc, Obb, Vec3 } from '@/api/types'

/** The 8 corners of an upright box: 4 at the bottom, then 4 at the top. */
export function obbCorners(obb: Obb): Vec3[] {
  const [cx, cy, cz] = obb.center
  const [L, W, H] = obb.size
  const c = Math.cos(obb.yaw)
  const s = Math.sin(obb.yaw)
  const ax = [(c * L) / 2, (s * L) / 2]
  const ay = [(-s * W) / 2, (c * W) / 2]
  const foot: [number, number][] = [
    [cx - ax[0] - ay[0], cy - ax[1] - ay[1]],
    [cx + ax[0] - ay[0], cy + ax[1] - ay[1]],
    [cx + ax[0] + ay[0], cy + ax[1] + ay[1]],
    [cx - ax[0] + ay[0], cy - ax[1] + ay[1]],
  ]
  const z0 = cz - H / 2
  const z1 = cz + H / 2
  return [...foot.map(([x, y]): Vec3 => [x, y, z0]), ...foot.map(([x, y]): Vec3 => [x, y, z1])]
}

/** Index pairs of the 12 edges of obbCorners(). */
export const BOX_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]

/** The 12 edges of a box as a flat list of segment end points. */
export function boxSegments(obb: Obb): Vec3[] {
  const c = obbCorners(obb)
  return BOX_EDGES.flatMap(([i, j]) => [c[i], c[j]])
}

function applyRigid(m: number[][], p: Vec3): Vec3 {
  return [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
  ]
}

export const cameraCentre = (cam: CameraDoc): Vec3 => [cam.cam2world[0][3], cam.cam2world[1][3], cam.cam2world[2][3]]

/** Line segments that draw a camera as a small pyramid, `depth` metres deep. */
export function frustumSegments(cam: CameraDoc, depth = 0.22): Vec3[] {
  const [H, W] = cam.size
  const fx = cam.K[0][0]
  const fy = cam.K[1][1]
  const cx = cam.K[0][2]
  const cy = cam.K[1][2]
  const corners = ([[0, 0], [W, 0], [W, H], [0, H]] as const).map(([u, v]) =>
    applyRigid(cam.cam2world, [((u - cx) / fx) * depth, ((v - cy) / fy) * depth, depth]))
  const apex = cameraCentre(cam)
  const out: Vec3[] = []
  for (let i = 0; i < 4; i++) {
    out.push(apex, corners[i], corners[i], corners[(i + 1) % 4])
  }
  return out
}

/** Project world points into a camera. Returns pixel (u, v) or null when behind it. */
export function projector(cam: CameraDoc): (p: Vec3) => [number, number] | null {
  // cam2world is rigid (rotation R, translation t): world2cam = [R^T | -R^T t]
  const m = cam.cam2world
  const t: Vec3 = [m[0][3], m[1][3], m[2][3]]
  return (p) => {
    const d: Vec3 = [p[0] - t[0], p[1] - t[1], p[2] - t[2]]
    const x = m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2]
    const y = m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2]
    const z = m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2]
    if (z < 0.05) return null
    return [(cam.K[0][0] * x) / z + cam.K[0][2], (cam.K[1][1] * y) / z + cam.K[1][2]]
  }
}

export const midpoint = (a: Vec3, b: Vec3): Vec3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
