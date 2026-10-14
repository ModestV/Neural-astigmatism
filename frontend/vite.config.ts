import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Адрес бэкенда для разработки; переопределяется переменной BACKEND_URL.
const backend = process.env.BACKEND_URL ?? 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': backend,
      '/docs': backend,
      '/openapi.json': backend,
    },
  },
})
