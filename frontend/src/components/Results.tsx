// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// How answers are shown. A number never appears without its error bar, and an
// answer that leans on generated geometry always says so.

import { CircleCheck, CircleHelp, CircleX, EyeOff } from 'lucide-react'
import type { ReactNode } from 'react'

import type { FitResult, Measurement, Verdict } from '@/api/types'
import { Meter, type Tone } from '@/components/ui'
import { interval95, percent, withSigma } from '@/lib/format'

export function GeneratedNote() {
  return (
    <p className="generated-note">
      <EyeOff size={14} /> Partly based on geometry no camera saw.
    </p>
  )
}

export function MeasurementCard({ m, title }: { m: Measurement; title?: string }) {
  const relative = Math.abs(m.value_m) > 1e-6 ? m.sigma_m / Math.abs(m.value_m) : null
  return (
    <div className="result">
      <span className="result-title">{title ?? m.what}</span>
      <strong className="result-value">{withSigma(m)}</strong>
      <span className="result-sub">
        95% interval: {interval95(m)}
        {relative !== null && relative < 10 ? ` · 1σ is ${percent(relative, 1)} of the value` : ''}
      </span>
      {m.relies_on_generated && <GeneratedNote />}
    </div>
  )
}

const VERDICT: Record<Verdict, { tone: Tone; text: string; icon: ReactNode }> = {
  yes: { tone: 'good', text: 'It fits', icon: <CircleCheck size={22} /> },
  no: { tone: 'bad', text: 'It does not fit', icon: <CircleX size={22} /> },
  uncertain: { tone: 'warn', text: 'Too close to call', icon: <CircleHelp size={22} /> },
}

export function FitCard({ fit, children }: { fit: FitResult; children?: ReactNode }) {
  const v = VERDICT[fit.verdict]
  return (
    <div className={`result verdict tone-${v.tone}`}>
      <span className="verdict-head">
        {v.icon}
        <strong>{v.text}</strong>
        <em>p = {fit.probability.toFixed(2)}</em>
      </span>
      <Meter value={fit.probability} tone={v.tone} label="Probability that it fits" />
      <span className="result-sub">Orientation: {fit.orientation}.</span>
      <ul className="margins">
        {Object.entries(fit.margins).map(([axis, m]) => (
          <li key={axis}>
            <span>{axis}</span>
            <strong className={m.value_m < 0 ? 'neg' : undefined}>
              {m.value_m < 0 ? 'short by ' : 'spare '}
              {(Math.abs(m.value_m) * 100).toFixed(1)} ± {(m.sigma_m * 100).toFixed(1)} cm
            </strong>
          </li>
        ))}
      </ul>
      {children}
      {fit.relies_on_generated && <GeneratedNote />}
    </div>
  )
}
