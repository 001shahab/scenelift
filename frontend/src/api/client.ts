// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import type {
  Axis, DemoRequest, DistanceResult, EditRequest, EvalResult, FitBetweenResult, FitResult,
  GeometryPayload, GroundTruthEntry, Job, Measurement, PlacementsResult, SceneDoc, SceneSummary,
  SystemInfo, ToolSpec, Vec3, WallDistanceResult,
} from './types'

/** An error returned by the API, with the server's own explanation. */
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function describe(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: [...], msg: "..."}]
    return detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string }
        const where = (item.loc ?? []).filter((x) => x !== 'body').join('.')
        return where ? `${where}: ${item.msg ?? ''}` : (item.msg ?? '')
      })
      .join('; ')
  }
  return fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'Cannot reach the SceneLift server. Is `python app.py` still running?')
  }
  if (!res.ok) {
    let detail: unknown
    try {
      detail = ((await res.json()) as { detail?: unknown }).detail
    } catch {
      detail = undefined
    }
    throw new ApiError(res.status, describe(detail, `${res.status} ${res.statusText}`))
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

const get = <T>(path: string) => request<T>(path)

const post = <T>(path: string, body: unknown) =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

const scenePath = (id: string) => `/api/scenes/${encodeURIComponent(id)}`

export const api = {
  system: () => get<SystemInfo>('/api/system'),
  scenes: () => get<{ scenes: SceneSummary[] }>('/api/scenes').then((r) => r.scenes),
  scene: (id: string) => get<SceneDoc>(scenePath(id)),
  geometry: (id: string) => get<GeometryPayload>(`${scenePath(id)}/geometry`),
  deleteScene: (id: string) => request<void>(scenePath(id), { method: 'DELETE' }),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),

  createDemo: (req: DemoRequest) => post<Job>('/api/scenes/demo', req),
  createReconstruction: (form: FormData) =>
    request<Job>('/api/scenes/reconstruct', { method: 'POST', body: form }),

  edit: (id: string, req: EditRequest) =>
    post<{ scene: SceneDoc; new_id?: string; support?: string }>(`${scenePath(id)}/edit`, req),
  calibrate: (id: string, body: { object: string; axis: Axis; true_m: number }) =>
    post<{ factor: number; scene: SceneDoc }>(`${scenePath(id)}/calibrate`, body),

  distance: (id: string, a: string, b: string, mode: 'surface' | 'center') =>
    post<DistanceResult>(`${scenePath(id)}/query/distance`, { a, b, mode }),
  distanceToWall: (id: string, object: string) =>
    post<WallDistanceResult | null>(`${scenePath(id)}/query/distance-to-wall`, { object }),
  fitsInSpace: (id: string, body: { object: string; space: Vec3; allow_rotation: boolean; clearance: number }) =>
    post<FitResult>(`${scenePath(id)}/query/fits-in-space`, body),
  fitsBetween: (id: string, body: { object: string; a: string; b: string; clearance: number }) =>
    post<FitBetweenResult>(`${scenePath(id)}/query/fits-between`, body),
  placements: (id: string, body: { object: string; clearance: number }) =>
    post<PlacementsResult>(`${scenePath(id)}/query/placements`, body),

  selectClick: (id: string, camera: number, u: number, v: number) =>
    post<{ object: string | null }>(`${scenePath(id)}/select/click`, { camera, u, v }),
  selectBox: (id: string, camera: number, box: [number, number, number, number]) =>
    post<{ objects: string[] }>(`${scenePath(id)}/select/box`, { camera, box }),
  selectText: (id: string, text: string) =>
    post<{ objects: string[] }>(`${scenePath(id)}/select/text`, { text }),

  tools: () => get<{ tools: ToolSpec[] }>('/api/tools').then((r) => r.tools),
  callTool: (id: string, name: string, args: Record<string, unknown>) =>
    post<{ name: string; arguments: Record<string, unknown>; result: unknown }>(
      `${scenePath(id)}/tools/call`, { name, arguments: args }),
  evaluate: (id: string, ground_truth: GroundTruthEntry[]) =>
    post<EvalResult>(`${scenePath(id)}/evaluate`, { ground_truth }),

  /** `stamp` (the scene's creation time) keeps a browser cache from showing another scene's frame. */
  frameUrl: (id: string, index: number, stamp: string | null) =>
    `${scenePath(id)}/frames/${index}?t=${encodeURIComponent(stamp ?? '')}`,
  exportUrl: (id: string, format: 'glb' | 'obj' | 'ply' | 'points' | 'zip', background = false) =>
    `${scenePath(id)}/export?format=${format}${background ? '&background=true' : ''}`,
}

export type { Measurement }
