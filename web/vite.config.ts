import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api/* is proxied to the FastAPI app. Going through a proxy rather than
// calling http://127.0.0.1:8000 directly keeps the browser on one origin, so
// the backend needs no CORS middleware added purely for this optional UI.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
