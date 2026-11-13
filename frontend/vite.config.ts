// Copyright 2026 3S Holding OÜ
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath, URL } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `npm run build` writes the compiled app into the Python package, so
// `python app.py` serves it with no Node.js at run time.
// `npm run dev` serves the app on :5173 and forwards /api to `python app.py`.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      // the one copy of the 3S Holding logo lives in /static/img
      '@brand': fileURLToPath(new URL('../static/img', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
    fs: { allow: ['..'] },
  },
  build: {
    outDir: '../scenelift/studio/static',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1600,
  },
})
