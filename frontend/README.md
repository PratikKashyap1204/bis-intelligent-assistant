# Frontend

Vite + React + TypeScript UI for the BIS Intelligent Assistant.

The browser talks to the existing FastAPI API:

- `POST /api/search/answer` with `{ "query": "..." }`
- `GET /health` for connection status

In local development, Vite proxies those paths to `http://127.0.0.1:8000`, so no backend CORS change is required.

## Setup

```bash
cd frontend
npm install
```

No frontend secrets or API keys. Optional `VITE_API_BASE_URL` is only needed if you are **not** using the Vite proxy (not recommended for local demo; the backend does not currently enable CORS).

## Run

Start the backend first (from `backend/`):

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir .
```

Then:

```bash
cd frontend
npm run dev
```

- UI: http://localhost:5173
- API (proxied): http://localhost:5173/api/search/answer → http://127.0.0.1:8000/api/search/answer

## Test / build

```bash
npm test
npm run build
```
