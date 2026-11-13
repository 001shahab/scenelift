// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Two tabs for people who build on scenelift: the tool interface for language
// models (scenelift.query.TOOLS) and the benchmark metrics (scenelift.eval).

import { ClipboardCopy, FlaskConical, Play } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { api } from '@/api/client'
import type { EvalResult, GroundTruthEntry, SceneDoc, ToolSpec } from '@/api/types'
import { Button, NumberField, Section, SelectField, TextField } from '@/components/ui'
import { percent } from '@/lib/format'
import { message, useStudio } from '@/state/store'

// ------------------------------------------------------------------ tools
const OBJECT_ARGS = new Set(['object', 'a', 'b'])

export function ToolsTab() {
  const scene = useStudio((s) => s.scene)!
  const notify = useStudio((s) => s.notify)
  const [tools, setTools] = useState<ToolSpec[]>([])
  const [name, setName] = useState('list_objects')
  const [args, setArgs] = useState<Record<string, unknown>>({})
  const [output, setOutput] = useState<string | null>(null)
  const [calling, setCalling] = useState(false)

  useEffect(() => {
    api.tools().then(setTools).catch((e: unknown) => notify('error', message(e)))
  }, [notify])
  useEffect(() => setOutput(null), [scene.revision, name])

  const tool = tools.find((t) => t.name === name)
  const ids = scene.objects.map((o) => ({ value: o.id, label: o.id }))
  const set = (key: string, value: unknown) => setArgs((a) => ({ ...a, [key]: value }))

  const pick = (next: string) => {
    setName(next)
    // numbers start at the value the form shows, so what is sent is what is seen
    const start: Record<string, unknown> = {}
    const spec = tools.find((t) => t.name === next)
    for (const [key, p] of Object.entries(spec?.parameters.properties ?? {})) {
      if (p.type === 'number') start[key] = 1
    }
    setArgs(start)
  }

  const call = async () => {
    if (!tool) return
    const sent: Record<string, unknown> = {}
    for (const key of Object.keys(tool.parameters.properties)) {
      if (args[key] !== undefined && args[key] !== '') sent[key] = args[key]
    }
    setCalling(true)
    try {
      const res = await api.callTool(scene.id, tool.name, sent)
      setOutput(JSON.stringify(res.result, null, 2))
    } catch (e) {
      setOutput(`Error: ${message(e)}`)
    } finally {
      setCalling(false)
    }
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(tools, null, 2))
      notify('success', 'Tool definitions copied.')
    } catch {
      notify('error', 'The browser did not allow copying.')
    }
  }

  return (
    <div className="tab-body">
      <Section title="Tools for language models"
               aside={<Button size="sm" variant="ghost" icon={<ClipboardCopy size={14} />} onClick={() => void copy()}>Copy JSON</Button>}>
        <p className="hint">
          A language model should never guess a distance. Give it these functions: it turns the question into a
          call, and the number comes from the reconstruction, with its error bar. Try a call here exactly as a
          model would make it.
        </p>
        <SelectField label="Tool" value={tool ? name : ''} placeholder="Loading" onChange={pick}
                     options={tools.map((t) => ({ value: t.name, label: t.name }))} />
        {tool && <p className="tool-text">{tool.description}</p>}
        {tool && Object.entries(tool.parameters.properties).map(([key, spec]) => {
          const required = tool.parameters.required?.includes(key)
          const label = `${key}${required ? '' : ' (optional)'}`
          if (spec.enum) {
            return <SelectField key={key} label={label} value={(args[key] as string) ?? ''} placeholder="Default"
                                onChange={(v) => set(key, v)} options={spec.enum.map((v) => ({ value: v, label: v }))} />
          }
          if (spec.type === 'number') {
            return <NumberField key={key} label={label} unit="m" step={0.05}
                                value={typeof args[key] === 'number' ? (args[key] as number) : 1}
                                onChange={(v) => set(key, v)} />
          }
          if (OBJECT_ARGS.has(key)) {
            return <SelectField key={key} label={label} value={(args[key] as string) ?? ''} placeholder="Choose"
                                onChange={(v) => set(key, v)} options={ids} />
          }
          return <TextField key={key} label={label} value={(args[key] as string) ?? ''} onChange={(v) => set(key, v)} />
        })}
        <Button block variant="primary" icon={<Play size={15} />} disabled={!tool || calling} onClick={() => void call()}>
          Call {name}
        </Button>
        {output !== null && <pre className="output">{output}</pre>}
      </Section>
    </div>
  )
}

// ------------------------------------------------------------------ evaluate
function template(scene: SceneDoc): GroundTruthEntry[] {
  const [first, second] = scene.objects
  const out: GroundTruthEntry[] = []
  if (first) {
    out.push({ type: 'dimension', object: first.id, axis: 'length', true_m: +first.obb.size[0].toFixed(2) })
    out.push({ type: 'fit', object: first.id, space: [2.0, 1.0, 2.0], true: true })
  }
  if (first && second) out.push({ type: 'distance', a: first.id, b: second.id, mode: 'surface', true_m: 0.5 })
  return out
}

const METRICS: { key: keyof EvalResult; name: string; text: string; show: (v: number) => string }[] = [
  { key: 'abs_rel_error', name: 'Mean error', text: 'mean |measured − true| / true', show: (v) => percent(v, 1) },
  { key: 'median_rel_error', name: 'Median error', text: 'median relative error', show: (v) => percent(v, 1) },
  { key: 'coverage_68', name: '68% coverage', text: 'truths inside 1σ. Honest bars give about 68%', show: (v) => percent(v) },
  { key: 'coverage_95', name: '95% coverage', text: 'truths inside 1.96σ. Honest bars give about 95%', show: (v) => percent(v) },
  { key: 'mean_z', name: 'Mean z', text: 'signed error in σ. Far from 0 means a shared scale bias', show: (v) => v.toFixed(2) },
  { key: 'fit_accuracy', name: 'Fit accuracy', text: 'yes or no fit questions answered correctly', show: (v) => percent(v) },
  { key: 'fit_brier', name: 'Fit Brier score', text: 'mean squared error of the fit probability. Lower is better', show: (v) => v.toFixed(3) },
]

export function EvaluateTab() {
  const scene = useStudio((s) => s.scene)!
  const initial = useMemo(() => JSON.stringify(scene.ground_truth ?? template(scene), null, 2), [scene.id]) // eslint-disable-line react-hooks/exhaustive-deps
  const [text, setText] = useState(initial)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    setText(initial)
    setResult(null)
    setError(null)
  }, [initial])

  const run = async () => {
    setError(null)
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (e) {
      setError(`Not valid JSON: ${message(e)}`)
      return
    }
    if (!Array.isArray(parsed)) {
      setError('The ground truth must be a JSON list.')
      return
    }
    setRunning(true)
    try {
      setResult(await api.evaluate(scene.id, parsed as GroundTruthEntry[]))
    } catch (e) {
      setResult(null)
      setError(message(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="tab-body">
      <Section title="Ground truth">
        <p className="hint">
          {scene.ground_truth
            ? 'This is the synthetic room, so the exact sizes are known. They are filled in below.'
            : 'Measure a few lengths with a tape and enter them here. The numbers below are placeholders: replace them with your own.'}
          {' '}Three kinds of entry: dimension, distance and fit.
        </p>
        <textarea className="input code" rows={12} spellCheck={false} value={text}
                  aria-label="Ground truth as JSON" onChange={(e) => setText(e.target.value)} />
        <Button block variant="primary" icon={<FlaskConical size={15} />} disabled={running} onClick={() => void run()}>
          Evaluate
        </Button>
        {error && <p className="error-text">{error}</p>}
      </Section>

      {result && (
        <Section title="Accuracy and calibration"
                 aside={<span className="aside">{result.n_lengths} lengths · {result.n_fits} fits</span>}>
          <dl className="metrics">
            {METRICS.filter((m) => typeof result[m.key] === 'number').map((m) => (
              <div key={m.key}>
                <dt>{m.name}</dt>
                <dd>{m.show(result[m.key] as number)}</dd>
                <p>{m.text}</p>
              </div>
            ))}
          </dl>
          <table className="eval-rows">
            <thead>
              <tr><th>Question</th><th>Truth</th><th>Scene says</th></tr>
            </thead>
            <tbody>
              {result.rows.map((r, i) => {
                const fit = r.type === 'fit'
                const question = r.type === 'dimension' ? `${String(r.object)} ${String(r.axis)}`
                  : r.type === 'distance' ? `${String(r.a)} to ${String(r.b)}`
                  : `${String(r.object)} fits ${JSON.stringify(r.space)}`
                return (
                  <tr key={i}>
                    <td>{question}</td>
                    <td>{fit ? (r.true ? 'yes' : 'no') : `${Number(r.true_m).toFixed(3)} m`}</td>
                    <td>
                      {fit
                        ? `${String(r.verdict)} (p = ${Number(r.probability).toFixed(2)})`
                        : `${Number(r.measured_m).toFixed(3)} ± ${Number(r.sigma_m).toFixed(3)} m`}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Section>
      )}
    </div>
  )
}
