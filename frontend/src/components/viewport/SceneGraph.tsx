// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Everything drawn inside the 3D canvas. All coordinates here are scenelift
// world coordinates (metres, Z up); <ZUp> turns them into three.js's Y-up frame.

import { Html, Line } from '@react-three/drei'
import { useThree, type ThreeEvent } from '@react-three/fiber'
import { useEffect, useMemo, type ReactNode, type RefObject } from 'react'
import * as THREE from 'three'

import type { CameraDoc, SceneDoc, SceneObjectDoc, Vec3 } from '@/api/types'
import type { Cloud, Mesh, SceneGeometry } from '@/lib/decode'
import { length } from '@/lib/format'
import { boxSegments, cameraCentre, frustumSegments, midpoint } from '@/lib/space'
import { useStudio, type ColorMode, type Overlay, type ViewSettings } from '@/state/store'

import { GENERATED, GOOD, OBSERVED, type ScenePalette } from './palette'

/** scenelift is Z-up, three.js is Y-up: world (x, y, z) is drawn at (x, z, -y). */
export function ZUp({ children }: { children: ReactNode }) {
  return <group rotation={[-Math.PI / 2, 0, 0]}>{children}</group>
}

const toThree = (p: Vec3) => new THREE.Vector3(p[0], p[2], -p[1])

/** The DOM node that holds every label and tag drawn over the canvas. It is
 * handed to drei's <Html> explicitly: left to itself, <Html> attaches to a
 * node that changes once the canvas has connected its events, and a label
 * mounted before that moment is left empty. */
type LabelLayer = RefObject<HTMLElement>

function useDisposable<T extends { dispose: () => void } | null>(make: () => T, deps: unknown[]): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const value = useMemo(make, deps)
  useEffect(() => () => value?.dispose(), [value])
  return value
}

// ------------------------------------------------------------------ geometry
function provenanceColors(observed: Uint8Array): Float32Array {
  const seen = new THREE.Color(OBSERVED)
  const made = new THREE.Color(GENERATED)
  const out = new Float32Array(observed.length * 3)
  for (let i = 0; i < observed.length; i++) {
    const c = observed[i] ? seen : made
    out[3 * i] = c.r
    out[3 * i + 1] = c.g
    out[3 * i + 2] = c.b
  }
  return out
}

function meshGeometry(mesh: Mesh, mode: ColorMode): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
  g.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
  const colors = mode === 'provenance' ? provenanceColors(mesh.observed) : mesh.colors
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  g.computeBoundingSphere()
  return g
}

function cloudGeometry(cloud: Cloud): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(cloud.positions, 3))
  g.setAttribute('color', new THREE.BufferAttribute(cloud.colors, 3))
  g.computeBoundingSphere()
  return g
}

function segmentsGeometry(points: Vec3[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3))
  return g
}

function quadGeometry(quad: Vec3[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(quad.flat()), 3))
  g.setIndex([0, 1, 2, 0, 2, 3])
  return g
}

const isFlat = (quad: Vec3[]) => quad.every((p) => p.every((x) => x === 0))

// ------------------------------------------------------------------ objects
interface ObjectNodeProps {
  doc: SceneObjectDoc
  mesh: Mesh | null
  measured: Cloud
  view: ViewSettings
  state: 'selected' | 'secondary' | 'hovered' | 'idle'
  labels: LabelLayer
}

function ObjectNode({ doc, mesh, measured, view, state, labels }: ObjectNodeProps) {
  const select = useStudio((s) => s.select)
  const setHovered = useStudio((s) => s.setHovered)

  const surface = useDisposable(() => (mesh ? meshGeometry(mesh, view.colorMode) : null), [mesh, view.colorMode])
  const points = useDisposable(() => cloudGeometry(measured), [measured])
  const box = useMemo(() => boxSegments(doc.obb), [doc.obb])

  const handlers = {
    onClick: (e: ThreeEvent<MouseEvent>) => {
      if (e.delta > 3) return // the end of an orbit drag, not a click
      e.stopPropagation()
      select(doc.id, e.nativeEvent.shiftKey)
    },
    onPointerOver: (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation()
      setHovered(doc.id)
    },
    onPointerOut: () => setHovered(null),
  }

  const showSurface = view.surface && surface !== null
  const accent = state === 'secondary' ? GENERATED : OBSERVED
  const [cx, cy, cz] = doc.obb.center
  const top = cz + doc.obb.size[2] / 2

  return (
    <group>
      {showSurface && (
        <mesh geometry={surface} {...handlers}>
          <meshStandardMaterial
            vertexColors flatShading roughness={0.9} metalness={0} side={THREE.DoubleSide}
            // lift the surface a little without shifting its hue: the outline and the label carry the accent
            emissive="#ffffff" emissiveIntensity={state === 'idle' ? 0 : state === 'hovered' ? 0.03 : 0.05}
          />
        </mesh>
      )}
      {!showSurface && (
        // with the surface hidden, the upright box stays clickable
        <mesh position={[cx, cy, cz]} rotation={[0, 0, doc.obb.yaw]} {...handlers}>
          <boxGeometry args={doc.obb.size} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
      {(view.measured || !showSurface) && (
        <points geometry={points}>
          {view.colorMode === 'provenance'
            ? <pointsMaterial size={0.012} sizeAttenuation color={OBSERVED} />
            : <pointsMaterial size={0.012} sizeAttenuation vertexColors />}
        </points>
      )}
      {(view.boxes || state !== 'idle') && (
        <Line points={box} segments color={state === 'idle' ? '#858585' : accent}
              lineWidth={state === 'idle' || state === 'hovered' ? 1.2 : 2.4}
              transparent opacity={state === 'idle' ? 0.7 : 1} />
      )}
      {view.labels && (
        <Html position={[cx, cy, top + 0.05]} center zIndexRange={[20, 0]} portal={labels}>
          <button
            type="button"
            className={`label3d is-${state}`}
            onClick={(e) => select(doc.id, e.shiftKey)}
            onPointerEnter={() => setHovered(doc.id)}
            onPointerLeave={() => setHovered(null)}
          >
            {doc.id}
          </button>
        </Html>
      )}
    </group>
  )
}

// ------------------------------------------------------------------ room
function Room({ doc, palette }: { doc: SceneDoc; palette: ScenePalette }) {
  const { layout, bounds } = doc
  const floor = useMemo<Vec3[]>(() => {
    if (!isFlat(layout.floor_quad)) return layout.floor_quad
    const [x0, y0] = bounds.min
    const [x1, y1] = bounds.max
    return [[x0 - 0.5, y0 - 0.5, 0], [x1 + 0.5, y0 - 0.5, 0], [x1 + 0.5, y1 + 0.5, 0], [x0 - 0.5, y1 + 0.5, 0]]
  }, [layout.floor_quad, bounds])

  const floorGeometry = useDisposable(() => quadGeometry(floor), [floor])

  // a metric grid on the floor: a line every 0.5 m, a stronger one every metre
  const [fine, strong] = useMemo(() => {
    const xs = floor.map((p) => p[0])
    const ys = floor.map((p) => p[1])
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    const lines: [Vec3[], Vec3[]] = [[], []]
    const z = 0.002
    for (let i = Math.ceil(x0 / 0.5); i * 0.5 <= x1; i++) {
      lines[i % 2 === 0 ? 1 : 0].push([i * 0.5, y0, z], [i * 0.5, y1, z])
    }
    for (let j = Math.ceil(y0 / 0.5); j * 0.5 <= y1; j++) {
      lines[j % 2 === 0 ? 1 : 0].push([x0, j * 0.5, z], [x1, j * 0.5, z])
    }
    return lines
  }, [floor])
  const fineGeometry = useDisposable(() => segmentsGeometry(fine), [fine])
  const strongGeometry = useDisposable(() => segmentsGeometry(strong), [strong])

  return (
    <group>
      <mesh geometry={floorGeometry}>
        <meshBasicMaterial color={palette.floor} transparent opacity={layout.floor_found ? 0.9 : 0.45}
                           side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <lineSegments geometry={fineGeometry}>
        <lineBasicMaterial color={palette.grid} />
      </lineSegments>
      <lineSegments geometry={strongGeometry}>
        <lineBasicMaterial color={palette.gridStrong} />
      </lineSegments>
      {layout.walls.map((w, i) => <WallQuad key={i} quad={w.quad} color={palette.wall} />)}
    </group>
  )
}

function WallQuad({ quad, color }: { quad: Vec3[]; color: string }) {
  const geometry = useDisposable(() => quadGeometry(quad), [quad])
  return (
    <group>
      <mesh geometry={geometry}>
        <meshBasicMaterial color={color} transparent opacity={0.22} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Line points={[...quad, quad[0]]} color={color} lineWidth={1.5} />
    </group>
  )
}

// ------------------------------------------------------------------ cameras
function Cameras({ cameras, color }: { cameras: CameraDoc[]; color: string }) {
  const activeFrame = useStudio((s) => s.activeFrame)
  const setActiveFrame = useStudio((s) => s.setActiveFrame)
  const geometry = useDisposable(() => segmentsGeometry(cameras.flatMap((c) => frustumSegments(c))), [cameras])
  return (
    <group>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color={color} transparent opacity={0.8} />
      </lineSegments>
      {cameras.map((c) => (
        <mesh
          key={c.index}
          position={cameraCentre(c)}
          onClick={(e) => {
            if (e.delta > 3) return
            e.stopPropagation()
            setActiveFrame(c.index)
          }}
        >
          <sphereGeometry args={[0.035, 12, 12]} />
          <meshBasicMaterial color={c.index === activeFrame ? OBSERVED : color} />
        </mesh>
      ))}
    </group>
  )
}

// ------------------------------------------------------------------ answers
function Answers({ overlay, color, labels }: { overlay: Overlay; color: string; labels: LabelLayer }) {
  const { segment, placements } = overlay
  return (
    <group>
      {segment && (
        <group>
          <Line points={[segment.a, segment.b]} color={color} lineWidth={2.5} depthTest={false} renderOrder={10} />
          {[segment.a, segment.b].map((p, i) => (
            <mesh key={i} position={p} renderOrder={10}>
              <sphereGeometry args={[0.018, 12, 12]} />
              <meshBasicMaterial color={color} depthTest={false} />
            </mesh>
          ))}
          <Html position={midpoint(segment.a, segment.b)} center zIndexRange={[30, 0]} portal={labels}>
            <span className="tag3d">{segment.text}</span>
          </Html>
        </group>
      )}
      {placements?.items.map((p, i) => {
        const [L, W] = placements.footprint
        const [hx, hy] = p.yaw_deg === 90 ? [W / 2, L / 2] : [L / 2, W / 2]
        const [x, y] = p.center_xy
        const z = 0.012
        const outline: Vec3[] = [
          [x - hx, y - hy, z], [x + hx, y - hy, z], [x + hx, y + hy, z], [x - hx, y + hy, z], [x - hx, y - hy, z],
        ]
        return (
          <group key={i}>
            <Line points={outline} color={GOOD} lineWidth={2.2} />
            <Html position={[x, y, z]} center zIndexRange={[30, 0]} portal={labels}>
              <span className="tag3d tag3d-good">{i + 1} · spare {length(p.spare_m)}</span>
            </Html>
          </group>
        )
      })}
    </group>
  )
}

// ------------------------------------------------------------------ camera rig
interface Controls {
  target: THREE.Vector3
  update: () => void
}

/** Frames the scene whenever `fitKey` changes (a scene opened, or "Frame" pressed). */
export function CameraRig({ doc }: { doc: SceneDoc }) {
  const fitKey = useStudio((s) => s.fitKey)
  const preset = useStudio((s) => s.cameraPreset)
  const camera = useThree((s) => s.camera)
  const controls = useThree((s) => s.controls) as unknown as Controls | null

  useEffect(() => {
    if (!controls) return
    const lo = toThree(doc.bounds.min)
    const hi = toThree(doc.bounds.max)
    const whole = Math.max(lo.distanceTo(hi) / 2, 0.5)
    let centre = lo.clone().add(hi).multiplyScalar(0.5)
    let radius = whole
    // Frame the objects, not the far corners of the room: the background
    // reaches well past them and would leave the objects small in the middle.
    if (doc.objects.length > 0 && preset !== 'top') {
      const box = new THREE.Box3()
      for (const o of doc.objects) {
        const reach = Math.hypot(o.obb.size[0], o.obb.size[1]) / 2
        const [x, y, z] = o.obb.center
        box.expandByPoint(toThree([x - reach, y - reach, z - o.obb.size[2] / 2]))
        box.expandByPoint(toThree([x + reach, y + reach, z + o.obb.size[2] / 2]))
      }
      centre = box.getCenter(new THREE.Vector3())
      radius = Math.min(whole, Math.max(box.getSize(new THREE.Vector3()).length() / 2 * 1.25, 0.5))
    }
    const offset = preset === 'top'
      ? new THREE.Vector3(0, 2.6 * radius, 0.001)
      : new THREE.Vector3(1.25, 1.05, 1.45).normalize().multiplyScalar(3.1 * radius)
    camera.position.copy(centre).add(offset)
    camera.near = Math.max(radius / 500, 0.01)
    camera.far = whole * 60
    camera.updateProjectionMatrix()
    controls.target.copy(centre)
    controls.update()
    // doc.bounds is read when the key changes; edits must not move the camera
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, preset, controls, camera])

  return null
}

// ------------------------------------------------------------------ scene
interface SceneGraphProps {
  doc: SceneDoc
  geometry: SceneGeometry
  palette: ScenePalette
  /** where labels and tags are attached: a node laid over the canvas */
  labels: RefObject<HTMLElement | null>
}

export function SceneGraph({ doc, geometry, palette, labels: labelLayer }: SceneGraphProps) {
  const labels = labelLayer as LabelLayer
  const view = useStudio((s) => s.view)
  const overlay = useStudio((s) => s.overlay)
  const selectedId = useStudio((s) => s.selectedId)
  const secondaryId = useStudio((s) => s.secondaryId)
  const hoveredId = useStudio((s) => s.hoveredId)
  const background = useDisposable(() => cloudGeometry(geometry.background), [geometry.background])

  return (
    <ZUp>
      {view.layout && <Room doc={doc} palette={palette} />}
      {view.background && geometry.background.count > 0 && (
        <points geometry={background}>
          <pointsMaterial size={0.02} sizeAttenuation vertexColors transparent opacity={0.85} />
        </points>
      )}
      {doc.objects.map((o) => {
        const g = geometry.objects[o.id]
        if (!g) return null // the geometry of a fresh edit is still on its way
        const state = o.id === selectedId ? 'selected'
          : o.id === secondaryId ? 'secondary'
          : o.id === hoveredId ? 'hovered' : 'idle'
        return <ObjectNode key={o.id} doc={o} mesh={g.mesh} measured={g.measured} view={view} state={state}
                           labels={labels} />
      })}
      {view.cameras && doc.cameras.length > 0 && <Cameras cameras={doc.cameras} color={palette.camera} />}
      <Answers overlay={overlay} color={palette.measure} labels={labels} />
      <axesHelper args={[0.4]} />
    </ZUp>
  )
}
