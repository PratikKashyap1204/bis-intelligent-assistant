# Backend

FastAPI backend for the BIS Intelligent Assistant.

## Setup

From the `backend/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --app-dir .
```

API docs: http://127.0.0.1:8000/docs  
Health check: http://127.0.0.1:8000/health

## Test

```bash
pytest
```
