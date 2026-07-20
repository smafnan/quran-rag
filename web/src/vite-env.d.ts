/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the API when the UI is hosted separately from the backend
   *  (e.g. Netlify + Render). Empty/unset means same-origin. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
