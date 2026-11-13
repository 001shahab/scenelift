// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { Download } from 'lucide-react'
import { useState } from 'react'

import { api } from '@/api/client'
import type { HistoryEntry, SceneDoc, Wall } from '@/api/types'
import { Badge, Rows, Section, Switch } from '@/components/ui'
import { percent, seconds, STAGE_TEXT, when } from '@/lib/format'
import { useStudio } from '@/state/store'

/** One bar per view: how much its depth had to be rescaled to agree with the others. */
function PerViewScale({ factors }: { factors: number[] }) {
  const deviations = factors.map((f) => f - 1)
  const peak = Math.max(0.005, ...deviations.map(Math.abs))
  return (
    <div className="scale-chart" role="img"
         aria-label={`Scale correction per view, from ${percent(Math.min(...deviations), 1)} to ${percent(Math.max(...deviations), 1)}`}>
      {deviations.map((d, i) => (
        <div key={i} className="scale-col" title={`View ${i + 1}: depth × ${factors[i].toFixed(4)}`}>
          <span className={d >= 0 ? 'up' : 'down'} style={{ height: `${(Math.abs(d) / peak) * 50}%` }} />
        </div>
      ))}
    </div>
  )
}

function describe(h: HistoryEntry): string {
  const id = typeof h.id === 'string' ? h.id : ''
  switch (h.op) {
    case 'remove':
      return `Removed ${id}`
    case 'translate': {
      const d = Array.isArray(h.delta) ? (h.delta as number[]).map((x) => x.toFixed(2)).join(', ') : ''
      return `Moved ${id} by (${d}) m`
    }
    case 'rotate':
      return `Turned ${id} by ${typeof h.yaw_deg === 'number' ? h.yaw_deg.toFixed(1) : '?'}°`
    case 'duplicate':
      return `Copied ${id} to ${String(h.new_id ?? '')}`
    case 'calibrate':
      return `Calibrated: scale × ${typeof h.factor === 'number' ? h.factor.toFixed(4) : '?'}`
    default:
      return h.op
  }
}

function Scale({ scene }: { scene: SceneDoc }) {
  const s = scene.scale
  return (
    <Section title="Scale" aside={<Badge tone={s.calibrated ? 'good' : 'brand'}>{s.calibrated ? 'calibrated' : 'from the model'}</Badge>}>
      <div className="headline">
        <strong>± {percent(s.sigma_rel, 1)}</strong>
        <span>1σ error of any length in this scene, before local noise</span>
      </div>
      {s.calibrated ? (
        <p className="hint">
          Fixed with a known length (scale × {s.global_factor.toFixed(4)}). The error is now that of the reference.
        </p>
      ) : (
        <>
          <Rows rows={[
            ['View-to-view spread', `${percent(s.sigma_consistency, 2)} (measured on this scene)`],
            ['Shared bias prior', `${percent(s.sigma_backbone, 1)} (cannot be seen in images)`],
            ['Views', String(s.n_views)],
          ]} />
          <p className="hint">
            σ<sub>rel</sub> = √(σ<sub>backbone</sub>² + (1.25 · σ<sub>consistency</sub> / √N)²). Select an object and
            calibrate with one known size to remove the shared bias.
          </p>
        </>
      )}
      {s.per_view.length > 1 && (
        <>
          <PerViewScale factors={s.per_view} />
          <p className="hint">Depth correction applied to each view so that all views agree.</p>
        </>
      )}
    </Section>
  )
}

function Exports({ scene }: { scene: SceneDoc }) {
  const [background, setBackground] = useState(false)
  const items: { format: 'glb' | 'obj' | 'ply' | 'points' | 'zip'; name: string; text: string }[] = [
    { format: 'glb', name: 'glTF binary (.glb)', text: 'One named node per object, plus floor and walls. Y-up. Blender, Unity, Unreal, the web.' },
    { format: 'obj', name: 'Wavefront (.obj)', text: 'Meshes, Z-up, metres.' },
    { format: 'ply', name: 'Mesh (.ply)', text: 'Meshes with vertex colours, Z-up, metres.' },
    { format: 'points', name: 'Measured points (.ply)', text: 'Every point a camera measured: objects and background.' },
    { format: 'zip', name: 'Scene folder (.zip)', text: 'scene.json, arrays and keyframes. Reload with Scene.load().' },
  ]
  return (
    <Section title="Export">
      <ul className="exports">
        {items.map((it) => (
          <li key={it.format}>
            <span>
              <strong>{it.name}</strong>
              <em>{it.text}</em>
            </span>
            <a className="btn btn-default btn-sm" download
               href={api.exportUrl(scene.id, it.format, it.format === 'glb' && background)}>
              <Download size={14} /><span>Download</span>
            </a>
          </li>
        ))}
      </ul>
      <Switch label="Include background points in the .glb" checked={background} onChange={setBackground} />
    </Section>
  )
}

/** How many walls, and whether they are square to the scene axes: the X and Y
 *  axes are turned onto the dominant wall, so every wall of a box-shaped room
 *  is. One that is not is a slanted wall or a plane that is not a wall. */
function wallSummary(walls: Wall[]): string {
  if (walls.length === 0) return 'none found'
  const square = walls.filter((w) => Math.max(Math.abs(w.normal[0]), Math.abs(w.normal[1])) > 0.985).length // within 10°
  if (square === walls.length) return `${walls.length}, square to the axes`
  return `${walls.length}, ${square} square to the axes`
}

export function SceneTab() {
  const scene = useStudio((s) => s.scene)!
  const timings = Object.entries(scene.timings).filter(([, t]) => typeof t === 'number')
  const total = timings.reduce((n, [, t]) => n + t, 0)
  const slowest = Math.max(1e-9, ...timings.map(([, t]) => t))

  return (
    <div className="tab-body">
      <Scale scene={scene} />

      <Section title="Room">
        <Rows rows={[
          ['Floor', scene.layout.floor_found ? 'found, at z = 0' : 'not found: gravity prior used'],
          ['Walls', wallSummary(scene.layout.walls)],
          ['Objects', String(scene.objects.length)],
          ['Keyframes', String(scene.cameras.length)],
          ['Units', `${scene.units}, ${scene.up_axis} up`],
        ]} />
      </Section>

      <Exports scene={scene} />

      {timings.length > 0 && (
        <Section title="Pipeline" aside={<span className="aside">{seconds(total)} in total</span>}>
          <ul className="timings">
            {timings.map(([name, t]) => (
              <li key={name}>
                <span>{STAGE_TEXT[name] ?? name}</span>
                <i><b style={{ width: `${(t / slowest) * 100}%` }} /></i>
                <em>{seconds(t)}</em>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="Edit history" aside={<span className="aside">{scene.history.length}</span>}>
        {scene.history.length === 0
          ? <p className="hint">No edits yet. Select an object to move, turn, copy or remove it.</p>
          : <ol className="history">{scene.history.map((h, i) => <li key={i}>{describe(h)}</li>)}</ol>}
      </Section>

      <Section title="About this scene">
        <Rows rows={[
          ['Folder', <code key="id">{scene.id}</code>],
          ['Made', when(scene.created_at) || 'unknown'],
          ['From', scene.inputs.length ? scene.inputs.slice(0, 3).join(', ') + (scene.inputs.length > 3 ? ` and ${scene.inputs.length - 3} more` : '') : scene.source],
        ]} />
        <details className="config">
          <summary>Every tunable number used ({Object.keys(scene.config).length})</summary>
          <pre>{JSON.stringify(scene.config, null, 1)}</pre>
        </details>
      </Section>
    </div>
  )
}
