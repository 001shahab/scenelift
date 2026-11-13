// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import logo from '@brand/logo.png'
import { BookOpen, Moon, Plus, Sun } from 'lucide-react'

import { Badge, Button, IconButton } from '@/components/ui'
import { percent } from '@/lib/format'
import { useStudio } from '@/state/store'

export function TopBar() {
  const system = useStudio((s) => s.system)
  const scene = useStudio((s) => s.scene)
  const theme = useStudio((s) => s.theme)
  const setTheme = useStudio((s) => s.setTheme)
  const closeScene = useStudio((s) => s.closeScene)
  const setNewSceneOpen = useStudio((s) => s.setNewSceneOpen)

  return (
    <header className="topbar">
      <button type="button" className="brand" onClick={closeScene} title="Back to the start page">
        <img src={logo} alt="3S Holding" width={34} height={34} />
        <span className="brand-name">
          <strong>SceneLift</strong> Studio
        </span>
      </button>
      <span className="tagline">Film it. Edit it in 3D.</span>

      {scene && (
        <div className="scene-chip" aria-label="Open scene">
          <strong>{scene.name}</strong>
          <Badge>{scene.objects.length} objects</Badge>
          <Badge>{scene.cameras.length} views</Badge>
          <Badge tone={scene.scale.calibrated ? 'good' : 'brand'}
                 title="1-sigma relative error of any length measured in this scene">
            scale ± {percent(scene.scale.sigma_rel, 1)}{scene.scale.calibrated ? ' · calibrated' : ''}
          </Badge>
        </div>
      )}

      <div className="topbar-end">
        {system && (
          <Badge tone={system.can_reconstruct ? 'good' : 'muted'}
                 title={system.can_reconstruct
                   ? 'MapAnything and SAM 3 are installed: real footage can be reconstructed.'
                   : 'The model backends are not installed. The synthetic room works without them.'}>
            {system.can_reconstruct ? 'Model backends ready' : 'Synthetic mode'}
          </Badge>
        )}
        <Button variant="primary" size="sm" icon={<Plus size={15} />} onClick={() => setNewSceneOpen(true)}>
          New scene
        </Button>
        <a className="icon-btn" href="/api/docs" target="_blank" rel="noreferrer" title="REST API documentation"
           aria-label="REST API documentation">
          <BookOpen size={17} />
        </a>
        <IconButton label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                    onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
        </IconButton>
      </div>
    </header>
  )
}
