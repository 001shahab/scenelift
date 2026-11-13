// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// The keyframes of the scene. The strip under the 3D view opens a frame; in the
// open frame a click runs scenelift.select.by_click and a dragged box runs
// by_box, so objects can be picked in the photo as well as in 3D.

import { ChevronLeft, ChevronRight, ImageOff } from 'lucide-react'
import {
  useEffect, useMemo, useRef, useState,
  type CSSProperties, type PointerEvent as ReactPointerEvent,
} from 'react'

import { api } from '@/api/client'
import type { CameraDoc, SceneDoc } from '@/api/types'
import { cx, IconButton, Modal, Spinner } from '@/components/ui'
import { BOX_EDGES, obbCorners, projector } from '@/lib/space'
import { message, useStudio } from '@/state/store'

const canShow = (c: CameraDoc) => c.has_frame || c.has_depth

export function Filmstrip() {
  const scene = useStudio((s) => s.scene)
  const activeFrame = useStudio((s) => s.activeFrame)
  const setActiveFrame = useStudio((s) => s.setActiveFrame)
  if (!scene || scene.cameras.length === 0) return null
  return (
    <div className="filmstrip" aria-label="Keyframes">
      <span className="filmstrip-title">
        Keyframes
        <em>click a frame, then click an object in it</em>
      </span>
      <ol>
        {scene.cameras.map((c) => (
          <li key={c.index}>
            <button type="button" className={cx('thumb', c.index === activeFrame && 'is-active')}
                    onClick={() => setActiveFrame(c.index)} title={`Frame ${c.index + 1}`}>
              {canShow(c)
                ? <img src={api.frameUrl(scene.id, c.index, scene.created_at)} alt={`Frame ${c.index + 1}`} loading="lazy" />
                : <ImageOff size={18} />}
              <span>{c.index + 1}</span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  )
}

interface Drag {
  x0: number
  y0: number
  x1: number
  y1: number
}

function FrameCanvas({ scene, camera }: { scene: SceneDoc; camera: CameraDoc }) {
  const selectedId = useStudio((s) => s.selectedId)
  const select = useStudio((s) => s.select)
  const notify = useStudio((s) => s.notify)
  const surface = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<Drag | null>(null)
  const [working, setWorking] = useState(false)
  const [H, W] = camera.size

  // the selected object's upright box, drawn into this camera
  const outline = useMemo(() => {
    const obj = scene.objects.find((o) => o.id === selectedId)
    if (!obj) return []
    const project = projector(camera)
    const corners = obbCorners(obj.obb).map(project)
    return BOX_EDGES.flatMap(([i, j]) => {
      const a = corners[i]
      const b = corners[j]
      return a && b ? [{ a, b }] : []
    })
  }, [scene.objects, selectedId, camera])

  /** pointer position in the camera's own pixels */
  const pixel = (e: ReactPointerEvent): [number, number] => {
    const r = surface.current!.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H]
  }

  const finish = async (d: Drag) => {
    const isClick = Math.hypot(d.x1 - d.x0, d.y1 - d.y0) < 0.012 * W
    setWorking(true)
    try {
      if (isClick) {
        const { object } = await api.selectClick(scene.id, camera.index, d.x1, d.y1)
        select(object)
        if (!object) notify('info', 'That pixel is background: no object there.')
      } else {
        const { objects } = await api.selectBox(scene.id, camera.index, [d.x0, d.y0, d.x1, d.y1])
        if (objects.length === 0) {
          notify('info', 'No object fills enough of that box.')
        } else {
          select(objects[0])
          if (objects.length > 1) notify('info', `In the box: ${objects.join(', ')}. Selected ${objects[0]}.`)
        }
      }
    } catch (e) {
      notify('error', message(e))
    } finally {
      setWorking(false)
    }
  }

  if (!canShow(camera)) {
    return (
      <div className="frame-missing">
        <ImageOff size={28} />
        <p>This scene was saved without frames and without depth, so there is no image to click in.</p>
      </div>
    )
  }

  return (
    <div
      ref={surface}
      className="frame-surface"
      style={{ aspectRatio: `${W} / ${H}`, '--ratio': W / H } as CSSProperties}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.currentTarget.setPointerCapture(e.pointerId)
        const [x, y] = pixel(e)
        setDrag({ x0: x, y0: y, x1: x, y1: y })
      }}
      onPointerMove={(e) => {
        if (!drag) return
        const [x, y] = pixel(e)
        setDrag({ ...drag, x1: x, y1: y })
      }}
      onPointerUp={() => {
        if (!drag) return
        setDrag(null)
        void finish(drag)
      }}
      onPointerCancel={() => setDrag(null)}
    >
      <img src={api.frameUrl(scene.id, camera.index, scene.created_at)} alt={`Frame ${camera.index + 1}`}
           draggable={false} />
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        {outline.map(({ a, b }, i) => (
          <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="frame-outline" />
        ))}
        {drag && (
          <rect className="frame-drag" x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)}
                width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)} />
        )}
      </svg>
      {working && <span className="frame-working"><Spinner small /></span>}
    </div>
  )
}

export function FrameViewer() {
  const scene = useStudio((s) => s.scene)
  const activeFrame = useStudio((s) => s.activeFrame)
  const setActiveFrame = useStudio((s) => s.setActiveFrame)
  const selectedId = useStudio((s) => s.selectedId)
  const count = scene?.cameras.length ?? 0

  useEffect(() => {
    if (activeFrame === null || count === 0) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') setActiveFrame((activeFrame + count - 1) % count)
      if (e.key === 'ArrowRight') setActiveFrame((activeFrame + 1) % count)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeFrame, count, setActiveFrame])

  if (!scene || activeFrame === null || !scene.cameras[activeFrame]) return null
  const camera = scene.cameras[activeFrame]

  return (
    <Modal wide title={`Frame ${activeFrame + 1} of ${count}`} onClose={() => setActiveFrame(null)}
           footer={
             <>
               <span className="hint">
                 Click an object to select it. Drag a box to select what is inside.
                 {selectedId ? ` Selected: ${selectedId}.` : ''}
                 {!camera.has_frame && camera.has_depth ? ' Showing depth: this scene has no stored photos.' : ''}
               </span>
               <span className="frame-nav">
                 <IconButton label="Previous frame" onClick={() => setActiveFrame((activeFrame + count - 1) % count)}>
                   <ChevronLeft size={18} />
                 </IconButton>
                 <IconButton label="Next frame" onClick={() => setActiveFrame((activeFrame + 1) % count)}>
                   <ChevronRight size={18} />
                 </IconButton>
               </span>
             </>
           }>
      <FrameCanvas key={camera.index} scene={scene} camera={camera} />
    </Modal>
  )
}
