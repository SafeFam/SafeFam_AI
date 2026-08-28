from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.middleware import BodySizeLimitMiddleware


def _build_client(max_body_bytes: int) -> TestClient:
    application = FastAPI()
    application.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=max_body_bytes,
    )

    @application.post("/echo")
    async def echo(payload: dict) -> dict:
        return {"ok": True}

    return TestClient(application)


def test_allows_body_within_limit():
    client = _build_client(max_body_bytes=1_000)

    response = client.post("/echo", json={"text": "hi"})

    assert response.status_code == 200


def test_rejects_body_over_limit_before_parsing():
    client = _build_client(max_body_bytes=50)

    response = client.post("/echo", json={"text": "x" * 500})

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


def test_multibyte_body_within_limit_passes():
    """멀티바이트(한글) 본문도 상한 이내면 그대로 통과한다."""
    client = _build_client(max_body_bytes=2_000)

    response = client.post("/echo", json={"text": "가" * 100})

    assert response.status_code == 200
