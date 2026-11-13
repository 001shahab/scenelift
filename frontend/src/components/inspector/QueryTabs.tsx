// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Spatial questions. Every answer is computed by scenelift.query on the
// server; this file only asks and shows.

import { MapPin, Move, Ruler } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { api } from '@/api/client'
import type { DistanceResult, FitBetweenResult, FitResult, Placement, PlacementsResult, SceneDoc } from '@/api/types'
import { FitCard, MeasurementCard } from '@/components/Results'
import { Button, Empty, NumberField, Section, Segmented, SelectField, Switch } from '@/components/ui'
import { length, withSigma } from '@/lib/format'
import { message, useStudio } from '@/state/store'

const objectOptions = (scene: SceneDoc, except: (string | null)[] = []) =>
  scene.objects.filter((o) => !except.includes(o.id)).map((o) => ({ value: o.id, label: o.id }))

/** Runs an API call and keeps its result, loading flag and error together. */
function useAsk<T>() {
  const notify = useStudio((s) => s.notify)
  const [result, setResult] = useState<T | null>(null)
  const [asking, setAsking] = useState(false)
  const ask = async (call: () => Promise<T>): Promise<T | null> => {
    setAsking(true)
    try {
      const r = await call()
      setResult(r)
      return r
    } catch (e) {
      setResult(null)
      notify('error', message(e))
      return null
    } finally {
      setAsking(false)
    }
  }
  return { result, asking, ask, clear: () => setResult(null) }
}

// ------------------------------------------------------------------ measure
export function MeasureTab() {
  const scene = useStudio((s) => s.scene)!
  const a = useStudio((s) => s.selectedId)
  const b = useStudio((s) => s.secondaryId)
  const select = useStudio((s) => s.select)
  const setSecondary = useStudio((s) => s.setSecondary)
  const setOverlay = useStudio((s) => s.setOverlay)
  const [mode, setMode] = useState<'surface' | 'center'>('surface')
  const [result, setResult] = useState<DistanceResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // measure as soon as two objects are chosen, and again after every edit
  useEffect(() => {
    setResult(null)
    setError(null)
    if (!a || !b || a === b) return
    let stale = false
    api.distance(scene.id, a, b, mode)
      .then((r) => {
        if (stale) return
        setResult(r)
        setOverlay({ segment: { a: r.segment[0], b: r.segment[1], text: withSigma(r) } })
      })
      .catch((e: unknown) => {
        if (!stale) setError(message(e))
      })
    return () => {
      stale = true
    }
  }, [scene.id, scene.revision, a, b, mode, setOverlay])

  if (scene.objects.length < 2) {
    return <Empty icon={<Ruler size={24} />} title="Two objects are needed">This scene has fewer than two objects.</Empty>
  }

  return (
    <div className="tab-body">
      <Section title="Distance between two objects">
        <p className="hint">
          Pick two objects here, or click one in the 3D view and shift-click the other.
        </p>
        <div className="grid-2">
          <SelectField label="From (A)" value={a ?? ''} placeholder="Choose" onChange={(id) => select(id)}
                       options={objectOptions(scene, [b])} />
          <SelectField label="To (B)" value={b ?? ''} placeholder="Choose" onChange={setSecondary}
                       options={objectOptions(scene, [a])} />
        </div>
        <Segmented
          label="What to measure"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'surface', label: 'Gap between surfaces', title: 'Closest points of the two surfaces' },
            { value: 'center', label: 'Centre to centre', title: 'Between the centres of the upright boxes' },
          ]}
        />
        {result && <MeasurementCard m={result} />}
        {error && <p className="error-text">{error}</p>}
        {!result && !error && a && b && <p className="hint">Measuring.</p>}
      </Section>
      <Section title="How the error bar is built">
        <p className="hint">
          σ² = (d · σ<sub>scale</sub>)² + 2 σ<sub>local</sub>² + Σ (k · h)². The first term is the scale error of
          this scene ({(100 * scene.scale.sigma_rel).toFixed(1)}%), which grows with the length. The second is surface
          noise at the two end points. The third appears only when an end point lies on generated geometry: h is its
          distance to the nearest measured point.
        </p>
      </Section>
    </div>
  )
}

// ------------------------------------------------------------------ fit
function FitInSpace({ scene }: { scene: SceneDoc }) {
  const selectedId = useStudio((s) => s.selectedId)
  const [object, setObject] = useState(selectedId ?? scene.objects[0].id)
  const [width, setWidth] = useState(1.0)
  const [depth, setDepth] = useState(0.6)
  const [height, setHeight] = useState(1.0)
  const [clearance, setClearance] = useState(0)
  const [rotate, setRotate] = useState(true)
  const { result, asking, ask, clear } = useAsk<FitResult>()

  useEffect(() => {
    if (selectedId) setObject(selectedId)
  }, [selectedId])
  useEffect(clear, [scene.revision, object]) // eslint-disable-line react-hooks/exhaustive-deps

  const o = scene.objects.find((x) => x.id === object)
  return (
    <Section title="Will it fit in a space?">
      <p className="hint">
        The space is taken as exactly known, from a tape measure or a product sheet, so the scene's full scale error
        applies to the object.
      </p>
      <SelectField label="Object" value={o ? object : ''} placeholder="Choose" onChange={setObject}
                   options={objectOptions(scene)} />
      <div className="grid-3">
        <NumberField label="Width" unit="m" value={width} min={0.01} step={0.05} onChange={setWidth} />
        <NumberField label="Depth" unit="m" value={depth} min={0.01} step={0.05} onChange={setDepth} />
        <NumberField label="Height" unit="m" value={height} min={0.01} step={0.05} onChange={setHeight} />
      </div>
      <div className="grid-2">
        <NumberField label="Clearance per axis" unit="m" value={clearance} min={0} step={0.01} onChange={setClearance} />
        <Switch label="May be turned by 90°" checked={rotate} onChange={setRotate} />
      </div>
      {o && (
        <p className="hint">
          {o.id} measures {length(o.obb.size[0])} × {length(o.obb.size[1])} × {length(o.obb.size[2])}.
        </p>
      )}
      <Button block variant="primary" disabled={!o || asking}
              onClick={() => void ask(() => api.fitsInSpace(scene.id, {
                object, space: [width, depth, height], allow_rotation: rotate, clearance }))}>
        Check the fit
      </Button>
      {result && <FitCard fit={result} />}
    </Section>
  )
}

function FitBetween({ scene }: { scene: SceneDoc }) {
  const setOverlay = useStudio((s) => s.setOverlay)
  const ids = scene.objects.map((o) => o.id)
  const [object, setObject] = useState(ids[0])
  const [a, setA] = useState(ids[1])
  const [b, setB] = useState(ids[2])
  const [clearance, setClearance] = useState(0)
  const { result, asking, ask, clear } = useAsk<FitBetweenResult>()
  useEffect(clear, [scene.revision, object, a, b]) // eslint-disable-line react-hooks/exhaustive-deps

  const valid = new Set([object, a, b]).size === 3 && [object, a, b].every((id) => ids.includes(id))
  const run = async () => {
    const r = await ask(() => api.fitsBetween(scene.id, { object, a, b, clearance }))
    if (r) setOverlay({ segment: { a: r.segment[0], b: r.segment[1], text: `gap ${withSigma(r.gap)}` } })
  }

  return (
    <Section title="Will it fit between two objects?">
      <p className="hint">
        Both lengths come from the same reconstruction, so a scale error stretches them together and only acts on
        their difference. This answer is tighter than comparing against an outside number.
      </p>
      <SelectField label="Object" value={object} onChange={setObject} options={objectOptions(scene)} />
      <div className="grid-2">
        <SelectField label="Between" value={a} onChange={setA} options={objectOptions(scene, [object])} />
        <SelectField label="And" value={b} onChange={setB} options={objectOptions(scene, [object])} />
      </div>
      <NumberField label="Clearance" unit="m" value={clearance} min={0} step={0.01} onChange={setClearance} />
      <Button block variant="primary" disabled={!valid || asking} onClick={() => void run()}>
        Check the gap
      </Button>
      {!valid && <p className="hint">Choose three different objects.</p>}
      {result && (
        <FitCard fit={result}>
          <span className="result-sub">The gap itself: {withSigma(result.gap)}.</span>
        </FitCard>
      )}
    </Section>
  )
}

function Placements({ scene }: { scene: SceneDoc }) {
  const selectedId = useStudio((s) => s.selectedId)
  const edit = useStudio((s) => s.edit)
  const busy = useStudio((s) => s.busy)
  const select = useStudio((s) => s.select)
  const setOverlay = useStudio((s) => s.setOverlay)
  const [object, setObject] = useState(selectedId ?? scene.objects[0].id)
  const [clearance, setClearance] = useState(0.02)
  const { result, asking, ask, clear } = useAsk<PlacementsResult>()

  useEffect(() => {
    if (selectedId) setObject(selectedId)
  }, [selectedId])
  useEffect(clear, [scene.revision, object]) // eslint-disable-line react-hooks/exhaustive-deps

  const exists = scene.objects.some((o) => o.id === object)

  const find = async () => {
    const r = await ask(() => api.placements(scene.id, { object, clearance }))
    if (r) setOverlay({ placements: { objectId: object, footprint: r.footprint, items: r.placements } })
  }

  /** Turn the object to the proposed orientation, then set it down on the proposed floor spot. */
  const moveTo = async (p: Placement) => {
    const before = scene.objects.find((o) => o.id === object)
    if (!before) return
    const turn = p.yaw_deg - (before.obb.yaw * 180) / Math.PI
    if (Math.abs(turn) > 0.5) {
      if (!(await edit({ op: 'rotate', object, yaw_deg: turn, settle: false }))) return
    }
    const now = useStudio.getState().scene?.objects.find((o) => o.id === object)
    if (!now) return
    const [cx, cy, cz] = now.obb.center
    const bottom = cz - now.obb.size[2] / 2
    await edit({ op: 'translate', object, delta: [p.center_xy[0] - cx, p.center_xy[1] - cy, -bottom], settle: true })
    select(object)
  }

  return (
    <Section title="Where could it stand?">
      <p className="hint">
        A top-down grid with 2 cm cells. A cell counts as free only if the floor there was actually observed and
        nothing stands on it. Floor that no camera saw is unknown, not free.
      </p>
      <div className="grid-2">
        <SelectField label="Object" value={exists ? object : ''} placeholder="Choose" onChange={setObject}
                     options={objectOptions(scene)} />
        <NumberField label="Clearance around it" unit="m" value={clearance} min={0} step={0.01} onChange={setClearance} />
      </div>
      <Button block variant="primary" icon={<MapPin size={15} />} disabled={!exists || asking} onClick={() => void find()}>
        Find free floor
      </Button>
      {result && result.placements.length === 0 && (
        <p className="hint">No free, observed floor area is large enough.</p>
      )}
      {result && result.placements.length > 0 && (
        <ol className="placements">
          {result.placements.map((p, i) => (
            <li key={i}>
              <span>
                <strong>x {p.center_xy[0].toFixed(2)}, y {p.center_xy[1].toFixed(2)} m</strong>
                <em>{p.yaw_deg === 90 ? 'long side along Y' : 'long side along X'} · spare {length(p.spare_m)}</em>
              </span>
              <Button size="sm" disabled={busy} icon={<Move size={14} />} onClick={() => void moveTo(p)}>
                Move here
              </Button>
            </li>
          ))}
        </ol>
      )}
    </Section>
  )
}

export function FitTab() {
  const scene = useStudio((s) => s.scene)!
  const thresholds = useMemo(() => {
    const yes = typeof scene.config.fit_yes === 'number' ? scene.config.fit_yes : 0.95
    const no = typeof scene.config.fit_no === 'number' ? scene.config.fit_no : 0.05
    return { yes, no }
  }, [scene.config])

  if (scene.objects.length === 0) {
    return <Empty icon={<Ruler size={24} />} title="No objects">There is nothing to fit in this scene.</Empty>
  }
  return (
    <div className="tab-body">
      <p className="hint">
        A fit answer is a probability, not a guess: the product over the axes of Φ(margin / σ). It reads "yes" at
        p ≥ {thresholds.yes}, "no" at p ≤ {thresholds.no}, and "uncertain" in between.
      </p>
      <FitInSpace scene={scene} />
      {scene.objects.length >= 3 && <FitBetween scene={scene} />}
      <Placements scene={scene} />
    </div>
  )
}
