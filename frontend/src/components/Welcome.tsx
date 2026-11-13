// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import logo from '@brand/logo.png'
import { Layers, Move3d, Ruler, Sparkles, UploadCloud } from 'lucide-react'

import { Button } from '@/components/ui'
import { useStudio } from '@/state/store'

const POINTS = [
  {
    icon: <Layers size={20} />,
    title: 'Measured and generated, kept apart',
    text: 'Every vertex is tagged as seen by a camera or filled in by a model. Switch the 3D view to Provenance to see which is which.',
  },
  {
    icon: <Ruler size={20} />,
    title: 'Every length has an error bar',
    text: 'Sizes, gaps and "will it fit" answers are computed on the reconstruction, with a 1-sigma uncertainty and a 95% interval.',
  },
  {
    icon: <Move3d size={20} />,
    title: 'Edit it like a real room',
    text: 'Move, turn, copy and delete objects. What stands on a table moves with it, and drops to the floor when the table goes.',
  },
]

export function Welcome() {
  const system = useStudio((s) => s.system)
  const startDemo = useStudio((s) => s.startDemo)
  const setNewSceneOpen = useStudio((s) => s.setNewSceneOpen)
  const hasScenes = useStudio((s) => s.scenes.length > 0)

  const quickDemo = () => {
    setNewSceneOpen(true)
    void startDemo({
      name: 'Synthetic room', n_views: 10, scale_jitter: 0.05, depth_noise: 0.003, seed: 0, completer: 'correct',
    })
  }

  return (
    <div className="welcome">
      <div className="welcome-inner">
        <img src={logo} alt="3S Holding" width={84} height={84} />
        <h1>Film it. Edit it in 3D.</h1>
        <p className="welcome-lead">
          SceneLift turns a short phone video, or a few photos, into a 3D scene in real metres where every
          object is separate. Delete, move and measure things, and ask whether something fits.
        </p>
        <div className="welcome-actions">
          <Button variant="primary" icon={<Sparkles size={17} />} onClick={quickDemo}>
            Try the synthetic room
          </Button>
          <Button icon={<UploadCloud size={17} />} onClick={() => setNewSceneOpen(true)}>
            Upload footage
          </Button>
        </div>
        <p className="hint">
          The synthetic room runs the whole geometry core on this computer in seconds. No GPU, no model weights.
          {system && !system.can_reconstruct && ' Real footage needs the model backends, which are not installed here.'}
          {hasScenes && ' Your saved scenes are in the list on the left.'}
        </p>
        <ul className="welcome-points">
          {POINTS.map((p) => (
            <li key={p.title}>
              <span className="point-icon">{p.icon}</span>
              <strong>{p.title}</strong>
              <p>{p.text}</p>
            </li>
          ))}
        </ul>
        <footer className="welcome-foot">
          {system && <>SceneLift {system.version} · </>}Apache-2.0 · © 2026 3S Holding OÜ · Designed and
          developed by Prof. Shahab Anbarjafari
        </footer>
      </div>
    </div>
  )
}
