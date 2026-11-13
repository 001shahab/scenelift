// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { Bot, Box, FlaskConical, Home, Ruler, ScanLine } from 'lucide-react'
import type { ReactNode } from 'react'

import { cx } from '@/components/ui'
import { useStudio, type TabId } from '@/state/store'

import { ObjectTab } from './ObjectTab'
import { FitTab, MeasureTab } from './QueryTabs'
import { SceneTab } from './SceneTab'
import { EvaluateTab, ToolsTab } from './ToolTabs'

const TABS: { id: TabId; label: string; title: string; icon: ReactNode }[] = [
  { id: 'scene', label: 'Scene', title: 'Scale, room, export, history', icon: <Home size={15} /> },
  { id: 'object', label: 'Object', title: 'Size, provenance, edits, calibration', icon: <Box size={15} /> },
  { id: 'measure', label: 'Measure', title: 'Distance between two objects', icon: <Ruler size={15} /> },
  { id: 'fit', label: 'Fit', title: 'Will it fit, and where', icon: <ScanLine size={15} /> },
  { id: 'tools', label: 'Tools', title: 'Function calling for language models', icon: <Bot size={15} /> },
  { id: 'evaluate', label: 'Evaluate', title: 'Accuracy and calibration against ground truth', icon: <FlaskConical size={15} /> },
]

export function Inspector() {
  const tab = useStudio((s) => s.tab)
  const setTab = useStudio((s) => s.setTab)
  return (
    <aside className="inspector">
      <div className="tabs" role="tablist" aria-label="Inspector">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={t.id === tab} title={t.title}
                  className={cx(t.id === tab && 'is-active')} onClick={() => setTab(t.id)}>
            {t.icon}
            <span>{t.label}</span>
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel">
        {tab === 'scene' && <SceneTab />}
        {tab === 'object' && <ObjectTab />}
        {tab === 'measure' && <MeasureTab />}
        {tab === 'fit' && <FitTab />}
        {tab === 'tools' && <ToolsTab />}
        {tab === 'evaluate' && <EvaluateTab />}
      </div>
    </aside>
  )
}
