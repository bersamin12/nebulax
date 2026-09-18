// Dev-only: build the transport-bar harness (transport-dev.html) on its own, as a compile
// check for web/src/state, web/src/api.js and web/src/components/transport/*.
// The real build (`npx vite build`, default config) is untouched and still ships index.html.
//   cd web && npx vite build --config dev/vite.transportdev.config.js --outDir <somewhere>
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export default defineConfig({
  root: WEB,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true, ws: true } },
  },
  build: {
    outDir: path.join(WEB, "dist-dev"),
    emptyOutDir: true,
    rollupOptions: { input: path.join(WEB, "transport-dev.html") },
  },
});
