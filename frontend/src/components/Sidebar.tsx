// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { AlertTriangle, Boxes, FolderOpen, Search, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import { api } from '@/api/client'
import type { SceneObjectDoc, SceneSummary } from '@/api/types'
import { Badge, cx, Empty, IconButton } from '@/components/ui'
import { flagText, metres, supportText, when } from '@/lib/format'
import { message, useStudio } from '@/state/store'

const SOURCE_TEXT: Record<SceneSummary['source'], string> = {
  demo: 'synthetic', upload: 'upload', cli: 'command line', studio: 'studio',
}

function SceneLibrary() {
  const scenes = useStudio((s) => s.scenes)
  const openId = useStudio((s) => s.scene?.id)
  const openScene = useStudio((s) => s.openScene)
  const deleteScene = useStudio((s) => s.deleteScene)

  if (scenes.length === 0) {
    return (
      <Empty icon={<FolderOpen size={22} />} title="No scenes yet">
        Build the synthetic room or upload footage with "New scene".
      </Empty>
    )
  }
  return (
    <ul className="scene-list">
      {scenes.map((s) => (
        <li key={s.id} className={cx(s.id === openId && 'is-active')}>
          <button type="button" className="scene-item" onClick={() => void openScene(s.id)}>
            <strong>{s.name}</strong>
            <span>
              {s.n_objects} objects · {s.n_views} views · {SOURCE_TEXT[s.source] ?? s.source}
            </span>
            <span>{when(s.created_at)}{s.calibrated ? ' · calibrated' : ''}</span>
          </button>
          <IconButton
            label={`Delete ${s.name}`}
            onClick={() => {
              if (window.confirm(`Delete the scene "${s.name}" from disk? This cannot be undone.`)) {
                void deleteScene(s.id)
              }
            }}
          >
            <Trash2 size={15} />
          </IconButton>
        </li>
      ))}
    </ul>
  )
}

function ObjectRow({ o }: { o: SceneObjectDoc }) {
  const selectedId = useStudio((s) => s.selectedId)
  const secondaryId = useStudio((s) => s.secondaryId)
  const select = useStudio((s) => s.select)
  const setHovered = useStudio((s) => s.setHovered)
  const [L, W, H] = o.obb.size
  const warn = o.flags.filter((f) => f === 'floating' || f.startsWith('overlaps:') || f.startsWith('completion'))
  return (
    <li>
      <button
        type="button"
        className={cx('object-item', o.id === selectedId && 'is-selected', o.id === secondaryId && 'is-secondary')}
        onClick={(e) => select(o.id, e.shiftKey)}
        onPointerEnter={() => setHovered(o.id)}
        onPointerLeave={() => setHovered(null)}
      >
        <span className="object-main">
          <strong>{o.id}</strong>
          {o.id === selectedId && <Badge tone="brand">A</Badge>}
          {o.id === secondaryId && <Badge tone="warn">B</Badge>}
          {warn.length > 0 && (
            <span className="object-warn" title={warn.map(flagText).join(', ')}>
              <AlertTriangle size={14} />
            </span>
          )}
        </span>
        <span className="object-sub">
          {metres(L)} × {metres(W)} × {metres(H)} · {supportText(o.support)}
        </span>
      </button>
    </li>
  )
}

function ObjectList() {
  const scene = useStudio((s) => s.scene)
  const select = useStudio((s) => s.select)
  const notify = useStudio((s) => s.notify)
  const [text, setText] = useState('')

  const shown = useMemo(() => {
    const t = text.trim().toLowerCase()
    if (!scene) return []
    return t ? scene.objects.filter((o) => o.id.toLowerCase().includes(t) || o.label.toLowerCase().includes(t))
             : scene.objects
  }, [scene, text])

  if (!scene) return null

  // Enter runs scenelift.select.by_text on the server: the same matching the library uses
  const byText = async () => {
    if (!text.trim()) return
    try {
      const { objects } = await api.selectText(scene.id, text.trim())
      if (objects.length === 0) {
        notify('info', `No object matches "${text.trim()}".`)
      } else {
        select(objects[0])
        if (objects.length > 1) notify('info', `Matched ${objects.join(', ')}. Selected ${objects[0]}.`)
      }
    } catch (e) {
      notify('error', message(e))
    }
  }

  return (
    <>
      <form className="search" role="search" onSubmit={(e) => { e.preventDefault(); void byText() }}>
        <Search size={15} aria-hidden="true" />
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Find by name, then Enter"
               aria-label="Find an object by name" />
      </form>
      {scene.objects.length === 0 ? (
        <Empty icon={<Boxes size={22} />} title="No objects">
          Nothing was separated from the background in this scene.
        </Empty>
      ) : shown.length === 0 ? (
        <p className="hint side-pad">Nothing matches "{text}".</p>
      ) : (
        <ul className="object-list">{shown.map((o) => <ObjectRow key={o.id} o={o} />)}</ul>
      )}
    </>
  )
}

export function Sidebar() {
  const scene = useStudio((s) => s.scene)
  const scenes = useStudio((s) => s.scenes)
  return (
    <aside className="sidebar">
      {scene && (
        <div className="side-block side-grow">
          <h2>Objects <span>{scene.objects.length}</span></h2>
          <ObjectList />
        </div>
      )}
      <div className={cx('side-block', !scene && 'side-grow')}>
        <h2>Scenes <span>{scenes.length}</span></h2>
        <SceneLibrary />
      </div>
    </aside>
  )
}
