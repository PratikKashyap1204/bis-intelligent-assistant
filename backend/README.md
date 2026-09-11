# Backend

FastAPI backend for the BIS Intelligent Assistant.

## Setup

From the `backend/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure `../.env` from `../.env.example`. Postgres with pgvector is
required (`DATABASE_URL`). Integration tests need `TEST_DATABASE_URL`
pointing at a separate `bis_test` database.

## Run

```bash
uvicorn app.main:app --reload --app-dir .
```

- API docs: http://127.0.0.1:8000/docs
- Health (includes a DB ping): http://127.0.0.1:8000/health
- Answer: `POST /api/search/answer`
- UI (separate Vite app): http://localhost:5173 — see `frontend/README.md`

Retrieval methods: `keyword`, `vector` (RAG default; M7 context eval
kept it), `hybrid`. Citations include `clause_id` (stable identity).

## Test

```bash
export TEST_DATABASE_URL="postgresql+psycopg2://bis_user:change_me@localhost:5432/bis_test"
pytest
```

No test makes a paid LLM or embedding-API call.
