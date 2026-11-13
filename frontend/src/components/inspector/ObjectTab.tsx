// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import {
  ArrowDown, ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUp, Copy, EyeOff, MousePointerClick,
  RotateCcw, RotateCw, Ruler, Trash2,
} from 'lucide-react'
import { useEffect, useState } from 'react'

import { api } from '@/api/client'
import type { Axis, SceneObjectDoc, Vec3, WallDistanceResult } from '@/api/types'
import { MeasurementCard } from '@/components/Results'
import { Badge, Button, Empty, Meter, NumberField, Rows, Section, SelectField, Switch } from '@/components/ui'
import { flagText, interval95, length, percent, supportText, withSigma } from '@/lib/format'
import { message, useSelectedObject, useStudio } from '@/state/store'

const AXES: Axis[] = ['length', 'width', 'height']

const COMPLETION_TEXT: Record<SceneObjectDoc['completion'], string> = {
  generated: 'A generated shape, registered to the measured points and accepted by the free-space and outline checks.',
  hull: 'The convex hull of the measured points. Nothing was invented.',
  none: 'Measured points only.',
}

function Sizes({ o }: { o: SceneObjectDoc }) {
  return (
    <table className="sizes">
      <tbody>
        {AXES.map((axis) => {
          const m = o.dimensions[axis]
          return (
            <tr key={axis}>
              <th scope="row">{axis}</th>
              <td>
                <strong>{withSigma(m)}</strong>
                <span>95%: {interval95(m)}</span>
              </td>
              <td>
                {m.relies_on_generated && (
                  <span className="unseen" title="More than a tenth of this extent was never seen by a camera">
                    <EyeOff size={14} />
                  </span>
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function Provenance({ o }: { o: SceneObjectDoc }) {
  const fit = Object.entries(o.fit)
  const fitText: Record<string, (v: number) => string> = {
    residual_m: (v) => `${length(v)} median distance from measured points to the shape`,
    freespace_violation: (v) => `${percent(v, 1)} of the shape sits where a camera saw through`,
    outline_iou: (v) => `${percent(v, 1)} overlap with the object's outline in the images`,
  }
  return (
    <>
      <div className="provenance">
        <Meter value={o.observed_fraction} label="Share of the surface that was observed" />
        <span>
          <strong>{percent(o.observed_fraction)}</strong> observed · {percent(1 - o.observed_fraction)} generated
        </span>
      </div>
      <p className="hint">{COMPLETION_TEXT[o.completion]}</p>
      {fit.length > 0 && (
        <ul className="plain">
          {fit.map(([k, v]) => <li key={k}>{fitText[k]?.(v) ?? `${k}: ${v}`}</li>)}
        </ul>
      )}
    </>
  )
}

function Mover({ o }: { o: SceneObjectDoc }) {
  const edit = useStudio((s) => s.edit)
  const busy = useStudio((s) => s.busy)
  const autoSettle = useStudio((s) => s.autoSettle)
  const setAutoSettle = useStudio((s) => s.setAutoSettle)
  const [step, setStep] = useState(0.1)
  const [angle, setAngle] = useState(15)
  const [carry, setCarry] = useState(true)

  const move = (delta: Vec3) => void edit({ op: 'translate', object: o.id, delta, carry })
  const turn = (yaw_deg: number) => void edit({ op: 'rotate', object: o.id, yaw_deg, carry })

  return (
    <>
      <div className="mover">
        <NumberField label="Step" unit="m" value={step} min={0.01} step={0.05} onChange={setStep} />
        <div className="pad" aria-label="Move the object">
          <Button size="sm" disabled={busy} title="Move along +Y" aria-label="Move along +Y"
                  icon={<ArrowUp size={15} />} onClick={() => move([0, step, 0])}>Y</Button>
          <Button size="sm" disabled={busy} title="Move along -X" aria-label="Move along -X"
                  icon={<ArrowLeft size={15} />} onClick={() => move([-step, 0, 0])}>X</Button>
          <Button size="sm" disabled={busy} title="Move along +X" aria-label="Move along +X"
                  icon={<ArrowRight size={15} />} onClick={() => move([step, 0, 0])}>X</Button>
          <Button size="sm" disabled={busy} title="Move along -Y" aria-label="Move along -Y"
                  icon={<ArrowDown size={15} />} onClick={() => move([0, -step, 0])}>Y</Button>
          <Button size="sm" disabled={busy} title="Raise" aria-label="Raise" onClick={() => move([0, 0, step])}>Z +</Button>
          <Button size="sm" disabled={busy} title="Lower" aria-label="Lower" onClick={() => move([0, 0, -step])}>Z −</Button>
        </div>
      </div>
      <div className="mover">
        <NumberField label="Turn" unit="°" value={angle} min={1} max={180} step={5} onChange={setAngle} />
        <div className="pad pad-turn" aria-label="Turn the object about the vertical axis">
          <Button size="sm" disabled={busy} title="Turn anticlockwise, seen from above"
                  icon={<RotateCcw size={15} />} onClick={() => turn(angle)} />
          <Button size="sm" disabled={busy} title="Turn clockwise, seen from above"
                  icon={<RotateCw size={15} />} onClick={() => turn(-angle)} />
          <Button size="sm" disabled={busy} onClick={() => turn(90)}>90°</Button>
        </div>
      </div>
      <Switch label="Carry what rests on it" checked={carry} onChange={setCarry}
              hint="A box on a table moves with the table." />
      <Switch label="Settle after each edit" checked={autoSettle} onChange={setAutoSettle}
              hint="Re-run support and gravity reasoning: close small gaps, push small overlaps apart, flag the rest." />
    </>
  )
}

function Calibrate({ o }: { o: SceneObjectDoc }) {
  const calibrate = useStudio((s) => s.calibrate)
  const notify = useStudio((s) => s.notify)
  const busy = useStudio((s) => s.busy)
  const [axis, setAxis] = useState<Axis>('height')
  const [trueM, setTrueM] = useState(+o.dimensions.height.value_m.toFixed(3))

  useEffect(() => {
    setTrueM(+o.dimensions[axis].value_m.toFixed(3))
  }, [o.id, axis, o.dimensions])

  const apply = async () => {
    const factor = await calibrate(o.id, axis, trueM)
    if (factor !== null) notify('success', `Scene rescaled by ${factor.toFixed(4)}. Lengths now carry your reference's error.`)
  }

  return (
    <>
      <p className="hint">
        Know one real size? Enter it. The whole scene is rescaled and the shared scale error, which no image can
        reveal, is replaced by the error of your tape measure.
      </p>
      <div className="grid-2">
        <SelectField label="Known size" value={axis} onChange={setAxis}
                     options={AXES.map((a) => ({ value: a, label: `${a} (now ${length(o.dimensions[a].value_m)})` }))} />
        <NumberField label="True length" unit="m" value={trueM} min={0.001} step={0.01} onChange={setTrueM} />
      </div>
      <Button block disabled={busy || !(trueM > 0)} icon={<Ruler size={15} />} onClick={() => void apply()}>
        Calibrate the scene
      </Button>
    </>
  )
}

export function ObjectTab() {
  const o = useSelectedObject()
  const scene = useStudio((s) => s.scene)
  const edit = useStudio((s) => s.edit)
  const select = useStudio((s) => s.select)
  const busy = useStudio((s) => s.busy)
  const notify = useStudio((s) => s.notify)
  const setOverlay = useStudio((s) => s.setOverlay)
  const [wall, setWall] = useState<WallDistanceResult | 'none' | null>(null)

  const key = o ? `${o.id}@${scene?.revision}` : ''
  useEffect(() => setWall(null), [key])

  if (!o || !scene) {
    return (
      <Empty icon={<MousePointerClick size={24} />} title="No object selected">
        Click an object in the 3D view, in a keyframe, or in the list on the left.
      </Empty>
    )
  }

  const [cx, cy, cz] = o.obb.center
  const warnings = o.flags.map(flagText)

  const duplicate = async () => {
    // put the copy beside the original, clear of its footprint, not inside it
    const [L, W] = o.obb.size
    const reach = Math.abs(Math.cos(o.obb.yaw)) * L + Math.abs(Math.sin(o.obb.yaw)) * W
    const res = await edit({ op: 'duplicate', object: o.id, offset: [+(reach + 0.1).toFixed(3), 0, 0] })
    if (res?.new_id) select(res.new_id)
  }

  const remove = async () => {
    const load = o.dependants.length ? ` ${o.dependants.join(', ')} will drop to the surface below.` : ''
    if (!window.confirm(`Remove ${o.id} from the scene?${load} The scene on disk is updated.`)) return
    await edit({ op: 'remove', object: o.id })
  }

  const toWall = async () => {
    try {
      const res = await api.distanceToWall(scene.id, o.id)
      setWall(res ?? 'none')
      if (res?.segment) setOverlay({ segment: { a: res.segment[0], b: res.segment[1], text: withSigma(res) } })
    } catch (e) {
      notify('error', message(e))
    }
  }

  return (
    <div className="tab-body">
      <header className="object-head">
        <h2>{o.id}</h2>
        <div className="badges">
          <Badge tone="brand">{o.label}</Badge>
          <Badge tone={o.support ? 'muted' : 'bad'}>{supportText(o.support)}</Badge>
          <Badge tone={o.completion === 'generated' ? 'warn' : 'muted'}>
            {o.completion === 'generated' ? 'generated shape' : o.completion === 'hull' ? 'measured hull' : 'points only'}
          </Badge>
          {warnings.map((w) => <Badge key={w} tone="bad">{w}</Badge>)}
        </div>
      </header>

      <Section title="Size">
        <Sizes o={o} />
      </Section>

      <Section title="Measured or generated">
        <Provenance o={o} />
      </Section>

      <Section title="Move and turn">
        <Mover o={o} />
      </Section>

      <Section title="Actions">
        <div className="actions">
          <Button disabled={busy} icon={<Copy size={15} />} onClick={() => void duplicate()}>Duplicate</Button>
          <Button disabled={busy} icon={<ArrowDownToLine size={15} />} title="Lower it onto whatever is underneath"
                  onClick={() => void edit({ op: 'drop', object: o.id })}>Drop</Button>
          <Button disabled={busy || scene.layout.walls.length === 0} icon={<Ruler size={15} />}
                  title={scene.layout.walls.length ? 'Gap to the nearest wall' : 'No walls were found in this scene'}
                  onClick={() => void toWall()}>Gap to wall</Button>
          <Button variant="danger" disabled={busy} icon={<Trash2 size={15} />} onClick={() => void remove()}>
            Remove
          </Button>
        </div>
        {wall === 'none' && <p className="hint">This scene has no walls.</p>}
        {wall && wall !== 'none' && <MeasurementCard m={wall} />}
      </Section>

      <Section title="Calibrate with a known size">
        <Calibrate o={o} />
      </Section>

      <Section title="Details">
        <Rows rows={[
          ['Centre', `x ${cx.toFixed(2)}, y ${cy.toFixed(2)}, z ${cz.toFixed(2)} m`],
          ['Turned', `${((o.obb.yaw * 180) / Math.PI).toFixed(1)}° about the vertical`],
          ['Rests', supportText(o.support)],
          ['Carries', o.dependants.length ? o.dependants.join(', ') : 'nothing'],
          ['Seen in', `${o.n_views} view${o.n_views === 1 ? '' : 's'}`],
          ['Measured points', o.n_points.toLocaleString()],
          ['Mesh', o.n_faces ? `${o.n_vertices.toLocaleString()} vertices, ${o.n_faces.toLocaleString()} faces` : 'none'],
        ]} />
      </Section>
    </div>
  )
}
