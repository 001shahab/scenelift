// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { AlertCircle, CircleCheck, Info, X } from 'lucide-react'
import { useEffect } from 'react'

import { Filmstrip, FrameViewer } from '@/components/Frames'
import { Inspector } from '@/components/inspector/Inspector'
import { NewScene } from '@/components/NewScene'
import { Sidebar } from '@/components/Sidebar'
import { TopBar } from '@/components/TopBar'
import { Button, Spinner } from '@/components/ui'
import { Viewport } from '@/components/viewport/Viewport'
import { Welcome } from '@/components/Welcome'
import { STAGE_TEXT } from '@/lib/format'
import { useStudio } from '@/state/store'

function Toasts() {
  const toasts = useStudio((s) => s.toasts)
  const dismiss = useStudio((s) => s.dismissToast)
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.kind}`} role={t.kind === 'error' ? 'alert' : 'status'}>
          {t.kind === 'error' ? <AlertCircle size={17} /> : t.kind === 'success' ? <CircleCheck size={17} /> : <Info size={17} />}
          <span>{t.text}</span>
          <button type="button" aria-label="Dismiss" onClick={() => dismiss(t.id)}><X size={15} /></button>
        </div>
      ))}
    </div>
  )
}

/** A running reconstruction stays visible after its dialog is closed. */
function JobPill() {
  const job = useStudio((s) => s.job)
  const open = useStudio((s) => s.newSceneOpen)
  const setOpen = useStudio((s) => s.setNewSceneOpen)
  if (!job || open) return null
  const running = job.stages.find((s) => s.status === 'running')
  return (
    <button type="button" className={`job-pill is-${job.status}`} onClick={() => setOpen(true)}>
      {job.status === 'error' ? <AlertCircle size={16} /> : <Spinner small />}
      <span>
        <strong>{job.name}</strong>
        {job.status === 'error' ? 'stopped. Click for details.'
          : running ? (STAGE_TEXT[running.name] ?? running.name) : 'waiting'}
      </span>
    </button>
  )
}

export default function App() {
  const booted = useStudio((s) => s.booted)
  const bootError = useStudio((s) => s.bootError)
  const boot = useStudio((s) => s.boot)
  const scene = useStudio((s) => s.scene)
  const loading = useStudio((s) => s.loadingScene)
  const newSceneOpen = useStudio((s) => s.newSceneOpen)

  useEffect(() => {
    void boot()
  }, [boot])

  if (!booted) {
    return <div className="splash"><Spinner /> Starting SceneLift Studio</div>
  }
  if (bootError) {
    return (
      <div className="splash splash-error">
        <AlertCircle size={28} />
        <strong>SceneLift Studio cannot reach its server.</strong>
        <p>{bootError}</p>
        <Button variant="primary" onClick={() => void boot()}>Try again</Button>
      </div>
    )
  }

  const showScene = scene !== null || loading
  return (
    <div className={`app ${showScene ? 'has-scene' : 'no-scene'}`}>
      <TopBar />
      <Sidebar />
      <main className="stage">
        {showScene ? (
          <>
            <Viewport />
            <Filmstrip />
          </>
        ) : (
          <Welcome />
        )}
      </main>
      {scene && <Inspector />}
      {newSceneOpen && <NewScene />}
      <FrameViewer />
      <JobPill />
      <Toasts />
    </div>
  )
}
