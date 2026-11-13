// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Shapes of the Studio REST API (scenelift/studio/app.py). Lengths are metres.
// The world is Z-up with the floor at z = 0.

export type Vec3 = [number, number, number]
export type Axis = 'length' | 'width' | 'height'
export type Verdict = 'yes' | 'no' | 'uncertain'

export interface SystemInfo {
  name: string
  version: string
  scenes_dir: string
  backends: { torch: boolean; mapanything: boolean; sam3: boolean; sam3d: boolean; sam3d_repo: string | null }
  can_reconstruct: boolean
  device: string | null
  stages: string[]
  default_prompts: string[]
  default_max_frames: number
  accepted: { images: string[]; videos: string[] }
  max_upload_mb: number
  license: string
  copyright: string
}

export interface SceneSummary {
  id: string
  name: string
  source: 'demo' | 'upload' | 'cli' | 'studio'
  created_at: string | null
  n_objects: number
  n_views: number
  calibrated: boolean
  labels: string[]
}

/** A length with its error bar: scenelift.query.Measurement. */
export interface Measurement {
  what: string
  value_m: number
  sigma_m: number
  interval95_m: [number, number]
  relies_on_generated: boolean
}

export interface DistanceResult extends Measurement {
  segment: [Vec3, Vec3]
}

export interface WallDistanceResult extends Measurement {
  wall?: number
  segment?: [Vec3, Vec3]
}

/** scenelift.query.FitResult */
export interface FitResult {
  verdict: Verdict
  probability: number
  orientation: string
  relies_on_generated: boolean
  margins: Record<string, Measurement>
}

export interface FitBetweenResult extends FitResult {
  gap: Measurement
  segment: [Vec3, Vec3]
}

export interface Placement {
  center_xy: [number, number]
  yaw_deg: number
  spare_m: number
}

export interface PlacementsResult {
  placements: Placement[]
  footprint: [number, number]
  height: number
}

export interface Obb {
  center: Vec3
  /** (length, width, height), length >= width */
  size: Vec3
  yaw: number
}

export interface SceneObjectDoc {
  id: string
  label: string
  completion: 'none' | 'hull' | 'generated'
  support: string | null
  flags: string[]
  n_views: number
  observed_fraction: number
  fit: Record<string, number>
  obb: Obb
  dimensions: Record<Axis, Measurement>
  n_points: number
  n_vertices: number
  n_faces: number
  dependants: string[]
}

export interface Wall {
  normal: Vec3
  d: number
  quad: [Vec3, Vec3, Vec3, Vec3]
}

export interface CameraDoc {
  index: number
  K: [Vec3, Vec3, Vec3]
  cam2world: number[][]
  /** (height, width) in pixels */
  size: [number, number]
  source: string | null
  has_frame: boolean
  has_depth: boolean
}

export interface ScaleInfo {
  per_view: number[]
  sigma_consistency: number
  sigma_backbone: number
  calibrated: boolean
  sigma_reference: number
  global_factor: number
  n_views: number
  sigma_rel: number
}

export interface HistoryEntry {
  op: string
  [key: string]: unknown
}

export type GroundTruthEntry = Record<string, unknown>

export interface SceneDoc {
  id: string
  name: string
  source: SceneSummary['source']
  created_at: string | null
  inputs: string[]
  params: Record<string, unknown>
  ground_truth: GroundTruthEntry[] | null
  revision: number
  units: string
  up_axis: string
  objects: SceneObjectDoc[]
  layout: { floor_found: boolean; floor_quad: [Vec3, Vec3, Vec3, Vec3]; walls: Wall[] }
  scale: ScaleInfo
  cameras: CameraDoc[]
  config: Record<string, unknown>
  timings: Record<string, number>
  history: HistoryEntry[]
  bounds: { min: Vec3; max: Vec3 }
}

export interface CloudPayload {
  n_points: number
  points: string
  colors: string
}

export interface MeshPayload {
  n_vertices: number
  n_faces: number
  vertices: string
  faces: string
  colors: string
  observed: string
}

export interface GeometryPayload {
  revision: number
  objects: Record<string, { measured: CloudPayload; mesh: MeshPayload | null }>
  background: CloudPayload
}

export type StageStatus = 'pending' | 'running' | 'done' | 'error'

export interface Job {
  id: string
  kind: 'demo' | 'reconstruct'
  name: string
  status: 'queued' | 'running' | 'done' | 'error'
  error: string | null
  scene_id: string | null
  created_at: string
  finished_at: string | null
  stages: { name: string; status: StageStatus; seconds: number | null }[]
}

export interface DemoRequest {
  name: string
  n_views: number
  scale_jitter: number
  depth_noise: number
  seed: number
  completer: 'correct' | 'wrong' | 'none'
}

export type EditOp = 'remove' | 'translate' | 'rotate' | 'duplicate' | 'drop' | 'settle'

export interface EditRequest {
  op: EditOp
  object?: string
  delta?: Vec3
  yaw_deg?: number
  offset?: Vec3
  carry?: boolean
  settle?: boolean
}

export interface ToolSpec {
  name: string
  description: string
  parameters: {
    type: string
    properties: Record<string, { type: string; enum?: string[] }>
    required?: string[]
  }
}

export interface EvalResult {
  n_lengths: number
  n_fits: number
  rows: Record<string, unknown>[]
  abs_rel_error?: number
  median_rel_error?: number
  coverage_68?: number
  coverage_95?: number
  mean_z?: number
  fit_accuracy?: number
  fit_brier?: number
}
