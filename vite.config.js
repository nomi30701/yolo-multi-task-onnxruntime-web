import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'
import mkcert from 'vite-plugin-mkcert'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    mkcert()
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  assetsInclude: ["**/*.onnx"],
  base: "/yolo-multi-task-onnxruntime-web/",

  server: {
    https: true
  },
})