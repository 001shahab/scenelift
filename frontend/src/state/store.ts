// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

// One store for the whole app. Components read what they need with selectors
// and call the actions below; only the actions talk to the API.

import { create } from 'zustand'

import { api, ApiError } from '@/api/client'
import type {
  Axis, DemoRequest, EditRequest, Job, Placement, SceneDoc, SceneSummary, SystemInfo, Vec3,
} from '@/api/types'
import { decodeGeometry, type SceneGeometry } from '@/lib/decode'

export type Theme = 'light' | 'dark'
export type TabId = 'scene' | 'object' | 'measure' | 'fit' | 'tools' | 'evaluate'
export type ColorMode = 'color' | 'provenance'

export interface ViewSettings {
  surface: boolean
  measured: boolean
  background: boolean
  layout: boolean
  boxes: boolean
  cameras: boolean
  labels: boolean
  colorMode: ColorMode
}

export interface Overlay {
  /** a measured gap drawn in the 3D view */
  segment: { a: Vec3; b: Vec3; text: string } | null
  /** free floor positions proposed for an object */
  placements: { objectId: string; footprint: [number, number]; items: Placement[] } | null
}

export interface Toast {
  id: number
  kind: 'info' | 'success' | 'error'
  text: string
}

const NO_OVERLAY: Overlay = { segment: null, placements: null }

function initialTheme(): Theme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

export const message = (e: unknown): string =>
  e instanceof ApiError || e instanceof Error ? e.message : String(e)

interface StudioState {
  booted: boolean
  bootError: string | null
  system: SystemInfo | null
  scenes: SceneSummary[]

  scene: SceneDoc | null
  geometry: SceneGeometry | null
  loadingScene: boolean
  busy: boolean

  selectedId: string | null
  /** the second object of a measurement (shift-click) */
  secondaryId: string | null
  hoveredId: string | null
  overlay: Overlay
  view: ViewSettings
  /** bumped to ask the 3D view to frame the scene again */
  fitKey: number
  cameraPreset: 'orbit' | 'top'
  /** index of the keyframe open in the frame viewer */
  activeFrame: number | null

  tab: TabId
  theme: Theme
  autoSettle: boolean
  newSceneOpen: boolean
  job: Job | null
  toasts: Toast[]

  boot: () => Promise<void>
  refreshScenes: () => Promise<void>
  openScene: (id: string) => Promise<void>
  closeScene: () => void
  deleteScene: (id: string) => Promise<void>
  startDemo: (req: DemoRequest) => Promise<void>
  startReconstruction: (form: FormData) => Promise<void>
  dismissJob: () => void

  edit: (req: EditRequest) => Promise<{ new_id?: string; support?: string } | null>
  calibrate: (object: string, axis: Axis, trueM: number) => Promise<number | null>

  select: (id: string | null, secondary?: boolean) => void
  setSecondary: (id: string | null) => void
  setHovered: (id: string | null) => void
  setOverlay: (overlay: Partial<Overlay>) => void
  setView: (view: Partial<ViewSettings>) => void
  frameScene: (preset?: 'orbit' | 'top') => void
  setActiveFrame: (index: number | null) => void
  setTab: (tab: TabId) => void
  setTheme: (theme: Theme) => void
  setAutoSettle: (on: boolean) => void
  setNewSceneOpen: (open: boolean) => void
  notify: (kind: Toast['kind'], text: string) => void
  dismissToast: (id: number) => void
}

let toastId = 0

export const useStudio = create<StudioState>()((set, get) => {
  /** Adopt the scene document returned by an edit, and reload the geometry if it moved. */
  const adopt = async (doc: SceneDoc) => {
    const ids = new Set(doc.objects.map((o) => o.id))
    const s = get()
    set({
      scene: doc,
      selectedId: s.selectedId && ids.has(s.selectedId) ? s.selectedId : null,
      secondaryId: s.secondaryId && ids.has(s.secondaryId) ? s.secondaryId : null,
      hoveredId: null,
      overlay: NO_OVERLAY,
    })
    if (get().geometry?.revision !== doc.revision) {
      const payload = await api.geometry(doc.id)
      if (get().scene?.id === doc.id) set({ geometry: decodeGeometry(payload) })
    }
  }

  const watch = async (jobId: string) => {
    if (get().job?.id !== jobId) return
    try {
      const job = await api.job(jobId)
      if (get().job?.id !== jobId) return
      set({ job })
      if (job.status === 'done') {
        await get().refreshScenes()
        if (job.scene_id) await get().openScene(job.scene_id)
        set({ job: null, newSceneOpen: false })
        get().notify('success', `"${job.name}" is ready.`)
        return
      }
      if (job.status === 'error') return // stays on screen until dismissed
      setTimeout(() => void watch(jobId), 500)
    } catch (e) {
      get().notify('error', message(e))
      setTimeout(() => void watch(jobId), 2500)
    }
  }

  return {
    booted: false,
    bootError: null,
    system: null,
    scenes: [],
    scene: null,
    geometry: null,
    loadingScene: false,
    busy: false,
    selectedId: null,
    secondaryId: null,
    hoveredId: null,
    overlay: NO_OVERLAY,
    view: {
      surface: true, measured: false, background: true, layout: true,
      boxes: false, cameras: true, labels: true, colorMode: 'color',
    },
    fitKey: 0,
    cameraPreset: 'orbit',
    activeFrame: null,
    tab: 'scene',
    theme: initialTheme(),
    autoSettle: true,
    newSceneOpen: false,
    job: null,
    toasts: [],

    boot: async () => {
      try {
        const [system, scenes] = await Promise.all([api.system(), api.scenes()])
        set({ system, scenes, booted: true, bootError: null })
      } catch (e) {
        set({ booted: true, bootError: message(e) })
      }
    },

    refreshScenes: async () => {
      try {
        set({ scenes: await api.scenes() })
      } catch (e) {
        get().notify('error', message(e))
      }
    },

    openScene: async (id) => {
      set({ loadingScene: true, selectedId: null, secondaryId: null, hoveredId: null,
            overlay: NO_OVERLAY, activeFrame: null })
      try {
        const [doc, payload] = await Promise.all([api.scene(id), api.geometry(id)])
        set((s) => ({ scene: doc, geometry: decodeGeometry(payload), loadingScene: false,
                      tab: 'scene', fitKey: s.fitKey + 1, cameraPreset: 'orbit' }))
      } catch (e) {
        set({ loadingScene: false })
        get().notify('error', message(e))
      }
    },

    closeScene: () =>
      set({ scene: null, geometry: null, selectedId: null, secondaryId: null, hoveredId: null,
            overlay: NO_OVERLAY, activeFrame: null }),

    deleteScene: async (id) => {
      try {
        await api.deleteScene(id)
        if (get().scene?.id === id) get().closeScene()
        await get().refreshScenes()
        get().notify('info', 'Scene deleted.')
      } catch (e) {
        get().notify('error', message(e))
      }
    },

    startDemo: async (req) => {
      try {
        const job = await api.createDemo(req)
        set({ job })
        void watch(job.id)
      } catch (e) {
        get().notify('error', message(e))
      }
    },

    startReconstruction: async (form) => {
      try {
        const job = await api.createReconstruction(form)
        set({ job })
        void watch(job.id)
      } catch (e) {
        get().notify('error', message(e))
      }
    },

    dismissJob: () => set({ job: null }),

    edit: async (req) => {
      const scene = get().scene
      if (!scene || get().busy) return null
      set({ busy: true })
      try {
        // removing an object always lets its load drop; other edits follow the switch
        const settle = req.settle ?? (req.op === 'remove' ? true : get().autoSettle)
        const { scene: doc, ...extra } = await api.edit(scene.id, { ...req, settle })
        await adopt(doc)
        void get().refreshScenes()
        return extra
      } catch (e) {
        get().notify('error', message(e))
        return null
      } finally {
        set({ busy: false })
      }
    },

    calibrate: async (object, axis, trueM) => {
      const scene = get().scene
      if (!scene || get().busy) return null
      set({ busy: true })
      try {
        const res = await api.calibrate(scene.id, { object, axis, true_m: trueM })
        await adopt(res.scene)
        set((s) => ({ fitKey: s.fitKey + 1 }))
        void get().refreshScenes()
        return res.factor
      } catch (e) {
        get().notify('error', message(e))
        return null
      } finally {
        set({ busy: false })
      }
    },

    select: (id, secondary = false) =>
      set((s) => {
        if (secondary) {
          if (!id || id === s.selectedId) return {}
          return s.selectedId ? { secondaryId: id, tab: 'measure' } : { selectedId: id }
        }
        return {
          selectedId: id,
          secondaryId: id && id === s.secondaryId ? null : s.secondaryId,
          tab: id && s.tab === 'scene' ? 'object' : s.tab,
        }
      }),

    setSecondary: (id) => set({ secondaryId: id }),
    setHovered: (id) => set((s) => (s.hoveredId === id ? {} : { hoveredId: id })),
    setOverlay: (overlay) => set((s) => ({ overlay: { ...s.overlay, ...overlay } })),
    setView: (view) => set((s) => ({ view: { ...s.view, ...view } })),
    frameScene: (preset = 'orbit') => set((s) => ({ fitKey: s.fitKey + 1, cameraPreset: preset })),
    setActiveFrame: (index) => set({ activeFrame: index }),
    setTab: (tab) => set({ tab }),

    setTheme: (theme) => {
      document.documentElement.dataset.theme = theme
      try {
        localStorage.setItem('scenelift.theme', theme)
      } catch {
        /* private browsing: the choice just does not persist */
      }
      set({ theme })
    },

    setAutoSettle: (on) => set({ autoSettle: on }),
    setNewSceneOpen: (open) => set({ newSceneOpen: open }),

    notify: (kind, text) => {
      const id = ++toastId
      set((s) => ({ toasts: [...s.toasts.slice(-3), { id, kind, text }] }))
      setTimeout(() => get().dismissToast(id), kind === 'error' ? 9000 : 4500)
    },

    dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  }
})

/** The selected object's document, or null. */
export const useSelectedObject = () =>
  useStudio((s) => s.scene?.objects.find((o) => o.id === s.selectedId) ?? null)
