// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { AlertCircle, Check, Circle, FileVideo, Images, Sparkles, UploadCloud, X } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'

import type { DemoRequest, Job } from '@/api/types'
import { Button, cx, Modal, NumberField, Segmented, Spinner, Switch, TextField } from '@/components/ui'
import { seconds, STAGE_TEXT } from '@/lib/format'
import { useStudio } from '@/state/store'

// ------------------------------------------------------------------ progress
export function JobProgress({ job }: { job: Job }) {
  const dismissJob = useStudio((s) => s.dismissJob)
  return (
    <div className="job">
      <ol className="steps">
        {job.stages.map((s) => (
          <li key={s.name} className={`step is-${s.status}`}>
            <span className="step-icon">
              {s.status === 'done' ? <Check size={14} />
                : s.status === 'running' ? <Spinner small />
                : s.status === 'error' ? <X size={14} />
                : <Circle size={10} />}
            </span>
            <span className="step-name">{STAGE_TEXT[s.name] ?? s.name}</span>
            <span className="step-time">{s.status === 'pending' ? '' : seconds(s.seconds)}</span>
          </li>
        ))}
      </ol>
      {job.status === 'queued' && <p className="hint">Waiting for the previous reconstruction to finish.</p>}
      {job.status === 'error' && (
        <div className="callout tone-bad" role="alert">
          <AlertCircle size={18} />
          <div>
            <strong>The reconstruction stopped.</strong>
            <p className="mono">{job.error}</p>
          </div>
          <Button size="sm" onClick={dismissJob}>Back</Button>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ synthetic room
const COMPLETER_TEXT: Record<DemoRequest['completer'], string> = {
  correct: 'The generator returns the right shape in a wrong pose and scale. Registration must recover both.',
  wrong: 'The generator returns a sphere for every object. The checks must reject it and fall back to the measured hull.',
  none: 'No generator at all. Every object is the convex hull of its measured points.',
}

function DemoForm() {
  const startDemo = useStudio((s) => s.startDemo)
  const [req, setReq] = useState<DemoRequest>({
    name: 'Synthetic room', n_views: 10, scale_jitter: 0.05, depth_noise: 0.003, seed: 0, completer: 'correct',
  })
  const patch = (p: Partial<DemoRequest>) => setReq((r) => ({ ...r, ...p }))
  return (
    <div className="form">
      <p className="lead">
        A table, a box standing on it, and a cabinet turned by 20 degrees, between two walls. Depth is ray-cast
        exactly, then each view gets its own scale error and pixel noise, which is how real depth models fail.
        The whole geometry core runs on your CPU in seconds, and the true sizes are stored for the Evaluate tab.
      </p>
      <TextField label="Name" value={req.name} onChange={(name) => patch({ name })} />
      <div className="grid-2">
        <NumberField label="Views" value={req.n_views} min={2} max={24} step={1}
                     onChange={(v) => patch({ n_views: Math.round(v) })} />
        <NumberField label="Random seed" value={req.seed} min={0} step={1}
                     onChange={(v) => patch({ seed: Math.max(0, Math.round(v)) })} />
        <NumberField label="Per-view scale error" unit="%" value={+(req.scale_jitter * 100).toFixed(2)}
                     min={0} max={20} step={1} onChange={(v) => patch({ scale_jitter: v / 100 })} />
        <NumberField label="Depth noise" unit="%" value={+(req.depth_noise * 100).toFixed(2)}
                     min={0} max={3} step={0.1} onChange={(v) => patch({ depth_noise: v / 100 })} />
      </div>
      <div className="field">
        <label>Shape generator</label>
        <Segmented
          label="Shape generator"
          value={req.completer}
          onChange={(completer) => patch({ completer })}
          options={[
            { value: 'correct', label: 'Right shape' },
            { value: 'wrong', label: 'Wrong shape' },
            { value: 'none', label: 'None' },
          ]}
        />
        <p className="hint">{COMPLETER_TEXT[req.completer]}</p>
      </div>
      <Button variant="primary" block icon={<Sparkles size={16} />}
              disabled={!req.name.trim() || req.n_views < 2 || req.n_views > 24}
              onClick={() => void startDemo({ ...req, name: req.name.trim() })}>
        Build the room
      </Button>
    </div>
  )
}

// ------------------------------------------------------------------ real footage
function UploadForm() {
  const system = useStudio((s) => s.system)
  const startReconstruction = useStudio((s) => s.startReconstruction)
  const notify = useStudio((s) => s.notify)
  const input = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [name, setName] = useState('')
  const [maxFrames, setMaxFrames] = useState(system?.default_max_frames ?? 24)
  const [prompts, setPrompts] = useState('')
  const [completion, setCompletion] = useState(true)
  const [over, setOver] = useState(false)

  if (!system) return null
  const { backends } = system
  const extOf = (f: File) => f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
  const isVideo = (f: File) => system.accepted.videos.includes(extOf(f))
  const isImage = (f: File) => system.accepted.images.includes(extOf(f))

  const take = (list: FileList | null) => {
    const picked = Array.from(list ?? [])
    const good = picked.filter((f) => isVideo(f) || isImage(f))
    if (good.length < picked.length) notify('error', 'Some files were skipped: unsupported type.')
    if (good.length === 0) return
    const video = good.find(isVideo)
    const next = video ? [video] : good
    if (video && good.length > 1) notify('info', 'One video at a time. The other files were left out.')
    setFiles(next)
    if (!name) setName(next[0].name.replace(/\.[^.]+$/, ''))
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    take(e.dataTransfer.files)
  }

  const submit = () => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f, f.name))
    form.append('name', name.trim() || 'Untitled scene')
    form.append('max_frames', String(maxFrames))
    if (prompts.trim()) form.append('prompts', prompts)
    form.append('completion', String(completion))
    void startReconstruction(form)
  }

  const megabytes = files.reduce((n, f) => n + f.size, 0) / 1e6

  return (
    <div className="form">
      {!system.can_reconstruct && (
        <div className="callout tone-warn">
          <AlertCircle size={18} />
          <div>
            <strong>The model backends are not installed here.</strong>
            <p>
              Real footage needs MapAnything {backends.mapanything ? '(found)' : '(missing)'} and
              SAM 3 {backends.sam3 ? '(found)' : '(missing)'}. Install them as described in
              <code> docs/BACKENDS.md</code>, then restart <code>python app.py</code>. The synthetic room
              needs none of this.
            </p>
          </div>
        </div>
      )}
      <div
        className={cx('dropzone', over && 'is-over')}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <UploadCloud size={26} />
        <strong>Drop a short video, or a few photos</strong>
        <span>15 to 30 seconds of walking around the scene works best. Up to {system.max_upload_mb} MB.</span>
        <input ref={input} type="file" multiple hidden
               accept={[...system.accepted.videos, ...system.accepted.images].join(',')}
               onChange={(e) => { take(e.target.files); e.target.value = '' }} />
      </div>
      {files.length > 0 && (
        <p className="file-summary">
          {isVideo(files[0]) ? <FileVideo size={15} /> : <Images size={15} />}
          {files.length === 1 ? files[0].name : `${files.length} photos`} · {megabytes.toFixed(1)} MB
          <button type="button" className="link" onClick={() => setFiles([])}>clear</button>
        </p>
      )}
      <div className="grid-2">
        <TextField label="Name" value={name} onChange={setName} placeholder="Living room" />
        <NumberField label="Keyframes at most" value={maxFrames} min={2} max={64} step={1}
                     onChange={(v) => setMaxFrames(Math.round(v))} />
      </div>
      <TextField label="Objects to look for" value={prompts} onChange={setPrompts}
                 placeholder={system.default_prompts.slice(0, 6).join(', ') + ', ...'}
                 hint={`Comma-separated. Leave empty for the ${system.default_prompts.length} default household nouns.`} />
      <Switch
        label={backends.sam3d ? 'Complete unseen parts with SAM 3D Objects' : 'Generative completion (not configured: objects get measured hulls)'}
        checked={completion && backends.sam3d}
        onChange={setCompletion}
      />
      <Button variant="primary" block icon={<UploadCloud size={16} />}
              disabled={!system.can_reconstruct || files.length === 0} onClick={submit}>
        Reconstruct
      </Button>
    </div>
  )
}

// ------------------------------------------------------------------ dialog
export function NewScene() {
  const job = useStudio((s) => s.job)
  const setNewSceneOpen = useStudio((s) => s.setNewSceneOpen)
  const [mode, setMode] = useState<'demo' | 'upload'>('demo')
  const running = job !== null

  return (
    <Modal
      title={running ? `Reconstructing "${job.name}"` : 'New scene'}
      onClose={() => setNewSceneOpen(false)}
    >
      {running ? (
        <JobProgress job={job} />
      ) : (
        <>
          <Segmented
            label="Source"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'demo', label: 'Synthetic room' },
              { value: 'upload', label: 'Your footage' },
            ]}
          />
          {mode === 'demo' ? <DemoForm /> : <UploadForm />}
        </>
      )}
    </Modal>
  )
}
