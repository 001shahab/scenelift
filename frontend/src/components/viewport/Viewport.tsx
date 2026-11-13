// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import {
  Box, Camera, CircleDot, Grid3x3, Layers, Maximize, ScanEye, SquareDashed, Tag, View,
} from 'lucide-react'
import { useRef, type ReactNode } from 'react'

import { IconButton, Segmented, Spinner } from '@/components/ui'
import { useStudio, type ViewSettings } from '@/state/store'

import { GENERATED, OBSERVED, PALETTE } from './palette'
import { CameraRig, SceneGraph } from './SceneGraph'

type Toggle = Exclude<keyof ViewSettings, 'colorMode'>

const TOGGLES: { key: Toggle; label: string; icon: ReactNode }[] = [
  { key: 'surface', label: 'Surfaces: the complete shape of every object', icon: <Box size={16} /> },
  { key: 'measured', label: 'Measured points: what the cameras actually saw', icon: <CircleDot size={16} /> },
  { key: 'background', label: 'Background points', icon: <Layers size={16} /> },
  { key: 'layout', label: 'Room: floor, walls and a 0.5 m grid', icon: <Grid3x3 size={16} /> },
  { key: 'boxes', label: 'Upright boxes used for sizes, support and fit', icon: <SquareDashed size={16} /> },
  { key: 'cameras', label: 'Cameras: click one to open its frame', icon: <Camera size={16} /> },
  { key: 'labels', label: 'Object labels', icon: <Tag size={16} /> },
]

export function Viewport() {
  const scene = useStudio((s) => s.scene)
  const geometry = useStudio((s) => s.geometry)
  const loading = useStudio((s) => s.loadingScene)
  const busy = useStudio((s) => s.busy)
  const view = useStudio((s) => s.view)
  const theme = useStudio((s) => s.theme)
  const setView = useStudio((s) => s.setView)
  const select = useStudio((s) => s.select)
  const frameScene = useStudio((s) => s.frameScene)
  const hoveredId = useStudio((s) => s.hoveredId)
  const palette = PALETTE[theme]
  const labels = useRef<HTMLDivElement>(null)

  return (
    <div className="viewport" style={{ cursor: hoveredId ? 'pointer' : undefined }}>
      {/* before the canvas in the tree, so the node exists when the 3D scene first renders */}
      <div ref={labels} className="viewport-labels" />
      <Canvas
        flat
        dpr={[1, 2]}
        camera={{ position: [5, 4, 5], fov: 42, near: 0.05, far: 500 }}
        onPointerMissed={() => select(null)}
      >
        <color attach="background" args={[palette.background]} />
        <ambientLight intensity={1.9} />
        <directionalLight position={[3, 8, 4]} intensity={1.9} />
        <directionalLight position={[-5, 3, -4]} intensity={0.6} />
        {scene && geometry && <SceneGraph doc={scene} geometry={geometry} palette={palette} labels={labels} />}
        {scene && <CameraRig doc={scene} />}
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.495} />
      </Canvas>

      <div className="viewport-bar" role="toolbar" aria-label="What the 3D view shows">
        <div className="tool-group">
          {TOGGLES.map((t) => (
            <IconButton key={t.key} label={t.label} active={view[t.key]}
                        onClick={() => setView({ [t.key]: !view[t.key] })}>
              {t.icon}
            </IconButton>
          ))}
        </div>
        <Segmented
          label="Colouring"
          value={view.colorMode}
          onChange={(colorMode) => setView({ colorMode })}
          options={[
            { value: 'color', label: 'Colour', title: 'Colours seen by the camera' },
            { value: 'provenance', label: 'Provenance', title: 'Observed versus generated geometry' },
          ]}
        />
        <div className="tool-group">
          <IconButton label="Frame the whole scene" onClick={() => frameScene('orbit')}><Maximize size={16} /></IconButton>
          <IconButton label="Look straight down" onClick={() => frameScene('top')}><View size={16} /></IconButton>
        </div>
      </div>

      {view.colorMode === 'provenance' && (
        <div className="legend">
          <span><i style={{ background: OBSERVED }} />Observed by a camera</span>
          <span><i style={{ background: GENERATED }} />Generated, never seen</span>
        </div>
      )}

      <p className="viewport-hint">
        <ScanEye size={14} /> Click selects. Shift-click picks a second object to measure. Drag to orbit, scroll to zoom.
      </p>

      {(loading || busy) && (
        <div className="viewport-busy"><Spinner small /> {loading ? 'Loading scene' : 'Updating'}</div>
      )}
    </div>
  )
}
