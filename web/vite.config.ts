import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build into ./dist with relative asset paths so FastAPI can serve it at /.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    // During `npm run dev`, proxy API calls to the FastAPI backend.
    proxy: { "/api": "http://localhost:8000" },
  },
});
