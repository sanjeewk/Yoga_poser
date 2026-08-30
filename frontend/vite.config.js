import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
  worker: {
    format: 'es',
  },
  server: {
    port: 5173,
    allowedHosts: ['.onamp.dev'],
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
