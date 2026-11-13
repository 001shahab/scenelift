// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import type { Measurement } from '@/api/types'

/** A length for people: centimetres below one metre, metres above. */
export function length(m: number): string {
  const a = Math.abs(m)
  if (a < 0.995) return `${(m * 100).toFixed(1)} cm`
  return `${m.toFixed(a < 10 ? 3 : 2)} m`
}

/** "85.2 cm ± 3.6" keeps the value and its 1-sigma error in the same unit. */
export function withSigma(m: Measurement): string {
  if (Math.abs(m.value_m) < 0.995) {
    return `${(m.value_m * 100).toFixed(1)} ± ${(m.sigma_m * 100).toFixed(1)} cm`
  }
  return `${m.value_m.toFixed(3)} ± ${m.sigma_m.toFixed(3)} m`
}

export function interval95(m: Measurement): string {
  const [lo, hi] = m.interval95_m
  return `${length(lo)} to ${length(hi)}`
}

export const metres = (m: number, digits = 2) => `${m.toFixed(digits)} m`

export const percent = (x: number, digits = 0) => `${(100 * x).toFixed(digits)}%`

export function seconds(s: number | null | undefined): string {
  if (s == null) return ''
  if (s < 1) return `${Math.round(s * 1000)} ms`
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
}

export function when(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const mins = Math.round((Date.now() - d.getTime()) / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  if (mins < 60 * 24) return `${Math.round(mins / 60)} h ago`
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

/** What an object rests on, in plain words. */
export function supportText(support: string | null): string {
  if (support === 'floor') return 'on the floor'
  if (support === 'wall') return 'on a wall'
  if (!support) return 'no support found'
  return `on ${support}`
}

/** A flag set by the pipeline, in plain words. */
export function flagText(flag: string): string {
  if (flag === 'floating') return 'floating'
  if (flag === 'completion_rejected') return 'generated shape rejected'
  if (flag.startsWith('overlaps:')) return `overlaps ${flag.slice(9)}`
  if (flag.startsWith('completion_error:')) return `completion failed (${flag.slice(17)})`
  return flag.replace(/_/g, ' ')
}

export const STAGE_TEXT: Record<string, string> = {
  models: 'Load models',
  ingest: 'Pick keyframes',
  geometry: 'Metric depth and cameras',
  scale: 'Harmonise scale',
  align: 'Find floor and walls',
  segment: 'Segment objects',
  associate: 'Link objects across views',
  fuse: 'Fuse measured points',
  complete: 'Complete and check shapes',
  physics: 'Supports and gravity',
  save: 'Save scene',
}
