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
