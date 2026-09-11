# BIS Intelligent Assistant

AI-powered assistant for questions about BIS Quality Control Orders and
related Indian Standard *metadata*, with source-backed answers.

This is an SIH project. The live corpus is a small **pilot**: freely
published QCO/circular PDFs plus standard catalogue metadata — **not**
the paid full text of Indian Standards.

## Current status (Milestones 1–8)

| Milestone | What shipped |
|---|---|
| M1 | Known-URL PDF ingestion, clause parsing, keyword retrieval |
| M2 | pgvector embeddings (`all-MiniLM-L6-v2`), semantic retrieval |
| M3 | Grounded RAG, citations, `POST /api/search/answer` |
| M4 | Optional LLM answers with extractive fallback |
| M5 | Corpus expansion (3 QCOs), 26-case retrieval evaluation |
| M6 | Hybrid RRF retrieval, vector cosine-similarity floor, eval-driven defaults |
| M7 | RAG/context eval, API hardening, stable `Clause.id` on re-ingest |
| M8 | React + TypeScript frontend for grounded answers and citations |

**Not included:** authentication, laboratory/product tables (schema
exists, tables are empty), Hindi generation, Docker files in this repo
(local Postgres uses `pgvector/pgvector:pg16`).

## Architecture

```
Frontend (Vite / React)
         → FastAPI POST /api/search/answer
known URL → fetcher → pdfplumber → clause parser → PostgreSQL
         → MiniLM embeddings (pgvector)
         → keyword | vector | hybrid retrieval
         → RAG context selection → extractive or optional LLM answer
         → citations from real Clause rows only (Clause.id is identity)
```

Public API:

- `GET /health` — process + PostgreSQL connectivity
- `POST /api/search/answer` — grounded answer (`method`: `vector` default,
  or `keyword` / `hybrid`)

Default `search()` with no method remains **keyword**. RAG's default
method remains **vector**: the M7 context evaluation showed vector
context-recall 0.474 vs keyword 0.158 at `max_context_items=5`. Hybrid
tied vector and was not selected.

## Live pilot corpus

- 5 standard **metadata** rows (IS 302 Part 1:2024; IS 3513 Parts 1–3:1989; IS 1475 Part 1:2001)
- 3 QCO/circular **documents** (no standard-body clause text)
- 144 clauses, 144 embeddings
- `clause_number` is **not** unique; `Clause.id` is the only stable identity

## Setup

```bash
cd bis-intelligent-assistant/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # edit DATABASE_URL if needed
```

Postgres must already be running with the `vector` extension (this
project's local container is `pgvector/pgvector:pg16`). Create tables:

```bash
python ../scripts/init_db.py
```

Ingest + embed (idempotent; unchanged documents keep `Clause.id`):

```bash
python ../scripts/ingest_pilot.py
python ../scripts/ingest_additional_standards.py
python ../scripts/embed_pilot.py
```

Run the API from `backend/`:

```bash
uvicorn app.main:app --reload --app-dir .
```

In a second terminal, run the UI from `frontend/`:

```bash
npm install
npm run dev
```

- UI: http://localhost:5173 (Vite proxies `/api` and `/health` to the API)
- Health: http://127.0.0.1:8000/health
- Docs: http://127.0.0.1:8000/docs

The frontend sends `{ "query": "..." }` to `POST /api/search/answer` and
renders the API's `answer`, `grounded`, `citations`, and `sources`. No
frontend API key is required. Do not set `VITE_API_BASE_URL` for local
development unless you have added CORS on the backend.

## Tests (offline, no paid API calls)

```bash
export TEST_DATABASE_URL="postgresql+psycopg2://bis_user:change_me@localhost:5432/bis_test"
pytest
```

Frontend (from `frontend/`):

```bash
npm test
```

## Evaluation (read-only against `bis_db`)

```bash
python ../scripts/run_retrieval_eval.py --save m7
python ../scripts/run_rag_eval.py --save m7_rag
```

## Answer generation

Default `ANSWER_PROVIDER=extractive` (deterministic, no API key). Set
`ANSWER_PROVIDER=llm` plus `OPENAI_API_KEY` in **local** `.env` for LLM
answers; missing key or LLM errors fall back to extractive. Never commit
keys. Manual paid demo only: `python scripts/llm_rag_demo.py`.
