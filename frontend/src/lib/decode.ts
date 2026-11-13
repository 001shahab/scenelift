// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// The geometry endpoint sends typed arrays as base64 (little-endian float32,
// uint32, uint8). This turns them into arrays WebGL can use directly.

import type { CloudPayload, GeometryPayload, MeshPayload } from '@/api/types'

export interface Cloud {
  count: number
  positions: Float32Array
  /** linear RGB, 3 floats per point */
  colors: Float32Array
}

export interface Mesh {
  vertexCount: number
  faceCount: number
  positions: Float32Array
  indices: Uint32Array
  /** linear RGB, 3 floats per vertex: the colour seen by the camera where observed */
  colors: Float32Array
  /** 1 where the vertex is close to a measured point, 0 where a model filled it in */
  observed: Uint8Array
}

export interface SceneGeometry {
  revision: number
  objects: Record<string, { measured: Cloud; mesh: Mesh | null }>
  background: Cloud
}

function bytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

const float32 = (b64: string) => {
  const b = bytes(b64)
  return new Float32Array(b.buffer, 0, Math.floor(b.byteLength / 4))
}

const uint32 = (b64: string) => {
  const b = bytes(b64)
  return new Uint32Array(b.buffer, 0, Math.floor(b.byteLength / 4))
}

// Colours are stored as 8-bit sRGB; the renderer works in linear light.
const SRGB_TO_LINEAR = new Float32Array(256)
for (let i = 0; i < 256; i++) {
  const c = i / 255
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function linearColors(b64: string): Float32Array {
  const raw = bytes(b64)
  const out = new Float32Array(raw.length)
  for (let i = 0; i < raw.length; i++) out[i] = SRGB_TO_LINEAR[raw[i]]
  return out
}

function cloud(p: CloudPayload): Cloud {
  return { count: p.n_points, positions: float32(p.points), colors: linearColors(p.colors) }
}

function mesh(p: MeshPayload): Mesh {
  return {
    vertexCount: p.n_vertices,
    faceCount: p.n_faces,
    positions: float32(p.vertices),
    indices: uint32(p.faces),
    colors: linearColors(p.colors),
    observed: bytes(p.observed),
  }
}

export function decodeGeometry(payload: GeometryPayload): SceneGeometry {
  const objects: SceneGeometry['objects'] = {}
  for (const [id, g] of Object.entries(payload.objects)) {
    objects[id] = { measured: cloud(g.measured), mesh: g.mesh ? mesh(g.mesh) : null }
  }
  return { revision: payload.revision, objects, background: cloud(payload.background) }
}
