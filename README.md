# BIS Intelligent Assistant

AI-powered Intelligent Assistant for Indian Standards and BIS Services for Industries and Consumers.

This SIH project will help users ask natural-language questions about BIS standards, product applicability, certification/testing guidance, and related laboratories — with source-backed answers in English and Hindi.

## Current status

**Foundation only.** This stage provides a minimal FastAPI backend skeleton, project layout, and a health check. No AI, RAG, database, scraper, or frontend yet.

## Architecture (current)

```
bis-intelligent-assistant/
├── backend/          # FastAPI application
│   ├── app/          # Application package
│   │   ├── api/      # Route modules (future)
│   │   ├── models/   # Schemas / data models (future)
│   │   ├── services/ # Business logic (future)
│   │   └── utils/    # Shared helpers (future)
│   └── tests/        # Pytest suite
├── data/             # Local data directories (raw / processed / metadata)
├── scripts/          # Utility scripts (future)
├── .env.example      # Sample environment variables
└── README.md
```

The backend exposes `GET /health` and returns `{"status": "ok"}`.

## Setup

### 1. Clone / enter the project

```bash
cd bis-intelligent-assistant
```

### 2. Create a virtual environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp ../.env.example ../.env
```

Edit `.env` as needed. No secrets are required for the foundation stage.

### 5. Start FastAPI

From the `backend/` directory (with the venv active):

```bash
uvicorn app.main:app --reload --app-dir .
```

- Health: http://127.0.0.1:8000/health
- Interactive docs: http://127.0.0.1:8000/docs

### 6. Run tests

From the `backend/` directory:

```bash
pytest
```

## What comes later

Modules under `api/`, `models/`, `services/`, and `data/` will grow as we add retrieval, standards matching, certification guidance, and multilingual support — still keeping the design simple and modular for a solo developer.

## RAG answer generation (Milestone 4)

`POST /api/search/answer` retrieves relevant BIS clauses and generates a
grounded answer with citations, via `RAGService` and a swappable
`AnswerGenerationProvider` (see `backend/app/services/answer_generation.py`
and `backend/app/services/rag.py`).

### Provider configuration

Set in `.env` (see `.env.example`):

```bash
# "extractive" (default): deterministic, offline, no API key, no network calls.
# "llm": real LLM-backed generation (OpenAI). Falls back to "extractive"
#        automatically on every request if OPENAI_API_KEY is not set.
ANSWER_PROVIDER=extractive

# Required only if ANSWER_PROVIDER=llm. Never commit a real key.
# OPENAI_API_KEY=sk-...

# Optional LLM tuning (defaults shown):
# LLM_MODEL_NAME=gpt-4o-mini
# LLM_TEMPERATURE=0.0
# LLM_MAX_OUTPUT_TOKENS=600
# LLM_TIMEOUT_SECONDS=20
```

Citation safety: the LLM only ever returns free-form answer text plus a
list of which `SOURCE_n` context items it used. `RAGService` independently
maps those ids back to the real retrieval results — the model can never
manufacture a clause number, page number, standard number, or URL that
isn't already in the database.

### Running the offline test suite

From `backend/` (with the venv active and `TEST_DATABASE_URL` set for the
PostgreSQL-backed integration tests):

```bash
pytest
```

No test makes a real LLM or network call — `LLMAnswerGenerationProvider`
is exercised via a fake client (see `backend/tests/test_llm_answer_generation.py`).

### Running the real LLM demo (manual, costs money, never run in CI)

```bash
export OPENAI_API_KEY=sk-...        # only in your shell or local .env — never commit it
python scripts/llm_rag_demo.py
```

This script refuses to run (and makes zero network calls) if
`OPENAI_API_KEY` is not set. When it does run, it makes at most 2 real LLM
calls: one in-scope pilot question and one deliberately out-of-scope
question, to demonstrate the system does not fabricate an answer when the
retrieved BIS material doesn't establish one.
