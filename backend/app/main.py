"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.search import router as search_router
from app.config import settings
from app.db.session import SessionLocal

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

app.include_router(search_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Avoid leaking internals on unexpected errors when DEBUG is off."""
    if isinstance(exc, (StarletteHTTPException, RequestValidationError)):
        raise exc
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    if settings.DEBUG:
        raise exc
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.get("/health")
def health() -> JSONResponse:
    """Liveness check that also verifies PostgreSQL connectivity."""
    database = "ok"
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception:
        logger.exception("Health check could not reach the database")
        database = "error"
    status = "ok" if database == "ok" else "error"
    code = 200 if status == "ok" else 503
    return JSONResponse({"status": status, "database": database}, status_code=code)
