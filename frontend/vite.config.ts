import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    proxy: {
      // Backend läuft im Docker-Container auf Port 8010
      "/api": { target: "http://localhost:8010", changeOrigin: true },
    },
  },
});
