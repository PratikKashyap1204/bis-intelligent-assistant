"""FastAPI application entry point."""

from fastapi import FastAPI

from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by tests and future deployments."""
    return {"status": "ok"}
