"""Tests for the health endpoint."""

from fastapi.testclient import TestClient


def test_health_ok_when_database_responds(monkeypatch) -> None:
    class _OkSession:
        def execute(self, _stmt):
            return None

        def close(self):
            return None

    monkeypatch.setattr("app.main.SessionLocal", lambda: _OkSession())
    from app.main import app

    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_503_when_database_unavailable(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("database unreachable")

    monkeypatch.setattr("app.main.SessionLocal", _boom)
    from app.main import app

    response = TestClient(app).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["database"] == "error"
