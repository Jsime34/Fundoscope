import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server (port 5180) forwards API calls to the FastAPI backend (port 8020),
// so the browser talks to one origin and no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8020",
      "/health": "http://127.0.0.1:8020",
    },
  },
});
