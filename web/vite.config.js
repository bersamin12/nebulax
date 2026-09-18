import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the FastAPI backend so the app is served from one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, ws: true },
    },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
