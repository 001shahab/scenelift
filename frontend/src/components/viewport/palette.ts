// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Colours of the 3D view. The brand colours are sampled from the 3S Holding
// logo: sky blue #8CC2FD, grey #858585, ink #2C232E, mist #E6E7E8.

import type { Theme } from '@/state/store'

export const BRAND = '#8CC2FD'
/** observed geometry, and the selected object */
export const OBSERVED = '#3D95EE'
/** generated geometry, and the second object of a measurement */
export const GENERATED = '#F2A33A'
export const GOOD = '#2FA873'

export interface ScenePalette {
  background: string
  floor: string
  grid: string
  gridStrong: string
  wall: string
  camera: string
  measure: string
}

export const PALETTE: Record<Theme, ScenePalette> = {
  light: {
    background: '#EEF0F3',
    floor: '#DFE3E8',
    grid: '#C9CFD7',
    gridStrong: '#AEB6C1',
    wall: '#B7C6D8',
    camera: '#858585',
    measure: '#2C232E',
  },
  dark: {
    background: '#1D171F',
    floor: '#2C232E',
    grid: '#3E3341',
    gridStrong: '#574A5B',
    wall: '#6C7E94',
    camera: '#A59FA8',
    measure: '#F3F3F5',
  },
}
