// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// Small building blocks shared by every panel. Styling lives in styles.css.

import { X } from 'lucide-react'
import {
  useEffect, useId, useState,
  type ButtonHTMLAttributes, type ReactNode,
} from 'react'

export type Tone = 'brand' | 'good' | 'warn' | 'bad' | 'muted'

const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(' ')

// ---------------------------------------------------------------- buttons
interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  size?: 'md' | 'sm'
  block?: boolean
  icon?: ReactNode
}

export function Button({ variant = 'default', size = 'md', block, icon, className, children, ...rest }: ButtonProps) {
  return (
    <button
      type="button"
      className={cx('btn', `btn-${variant}`, size === 'sm' && 'btn-sm', block && 'btn-block', className)}
      {...rest}
    >
      {icon}
      {children && <span>{children}</span>}
    </button>
  )
}

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string
  active?: boolean
}

export function IconButton({ label, active, className, children, ...rest }: IconButtonProps) {
  return (
    <button
      type="button"
      className={cx('icon-btn', active && 'is-active', className)}
      aria-label={label}
      aria-pressed={active}
      title={label}
      {...rest}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------- inputs
interface NumberFieldProps {
  label: string
  value: number
  onChange: (value: number) => void
  step?: number
  min?: number
  max?: number
  unit?: string
  disabled?: boolean
}

/** A number input that tolerates half-typed text ("-", "0.") without fighting the user. */
export function NumberField({ label, value, onChange, step = 0.1, min, max, unit, disabled }: NumberFieldProps) {
  const id = useId()
  const [text, setText] = useState(String(value))
  useEffect(() => {
    setText((t) => (Number(t) === value ? t : String(value)))
  }, [value])
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <div className="input-wrap">
        <input
          id={id}
          className="input"
          type="number"
          inputMode="decimal"
          value={text}
          step={step}
          min={min}
          max={max}
          disabled={disabled}
          onChange={(e) => {
            setText(e.target.value)
            const v = e.target.valueAsNumber
            if (Number.isFinite(v)) onChange(v)
          }}
        />
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  )
}

interface TextFieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  hint?: string
}

export function TextField({ label, value, onChange, placeholder, hint }: TextFieldProps) {
  const id = useId()
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} className="input" value={value} placeholder={placeholder}
             onChange={(e) => onChange(e.target.value)} />
      {hint && <p className="hint">{hint}</p>}
    </div>
  )
}

interface SelectFieldProps<T extends string> {
  label: string
  value: T | ''
  options: { value: T; label: string }[]
  onChange: (value: T) => void
  placeholder?: string
}

export function SelectField<T extends string>({ label, value, options, onChange, placeholder }: SelectFieldProps<T>) {
  const id = useId()
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <select id={id} className="input select" value={value} onChange={(e) => onChange(e.target.value as T)}>
        {placeholder !== undefined && <option value="" disabled>{placeholder}</option>}
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}

interface SwitchProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  hint?: string
}

export function Switch({ label, checked, onChange, hint }: SwitchProps) {
  return (
    <label className="switch" title={hint}>
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch-track" aria-hidden="true" />
      <span className="switch-label">{label}</span>
    </label>
  )
}

interface SegmentedProps<T extends string> {
  label: string
  value: T
  options: { value: T; label: string; title?: string }[]
  onChange: (value: T) => void
}

export function Segmented<T extends string>({ label, value, options, onChange }: SegmentedProps<T>) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} title={o.title}
                className={cx(o.value === value && 'is-active')} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- display
export function Badge({ tone = 'muted', children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return <span className={cx('badge', `tone-${tone}`)} title={title}>{children}</span>
}

export function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="section">
      <header className="section-head">
        <h3>{title}</h3>
        {aside}
      </header>
      {children}
    </section>
  )
}

export function Rows({ rows }: { rows: [label: string, value: ReactNode][] }) {
  return (
    <dl className="rows">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      {icon}
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  )
}

export const Spinner = ({ small }: { small?: boolean }) =>
  <span className={cx('spinner', small && 'spinner-sm')} role="status" aria-label="Working" />

/** A horizontal bar from 0 to 1. */
export function Meter({ value, tone = 'brand', label }: { value: number; tone?: Tone; label?: string }) {
  const clamped = Math.max(0, Math.min(1, value))
  return (
    <div className={cx('meter', `tone-${tone}`)} role="meter" aria-label={label}
         aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(clamped * 100)}>
      <span style={{ width: `${clamped * 100}%` }} />
    </div>
  )
}

// ---------------------------------------------------------------- modal
interface ModalProps {
  title: string
  onClose: () => void
  children: ReactNode
  wide?: boolean
  footer?: ReactNode
}

export function Modal({ title, onClose, children, wide, footer }: ModalProps) {
  const titleId = useId()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={cx('modal', wide && 'modal-wide')} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header className="modal-head">
          <h2 id={titleId}>{title}</h2>
          <IconButton label="Close" onClick={onClose}><X size={18} /></IconButton>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-foot">{footer}</footer>}
      </div>
    </div>
  )
}

export { cx }
