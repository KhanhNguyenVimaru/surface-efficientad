# EfficientAD Frontend (Nuxt 4 + Nuxt UI)

This folder contains the frontend for the FastAPI backend.

## Requirements

- Node.js 20+
- Backend running at `http://127.0.0.1:8000` (default)

## Configure API URL

If backend URL is different, create `.env` in `web/`:

```bash
NUXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

## Development

```bash
npm install
npm run dev
```

App URL:
`http://localhost:3000`

## Production Build

```bash
npm run build
npm run preview
```
