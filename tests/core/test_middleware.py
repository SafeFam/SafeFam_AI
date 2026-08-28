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
<<<<<<< HEAD
=======


# --- ASGI 레벨: Content-Length 헤더 없이 다중 프레임으로 오는 요청 방어 ---


async def _echo_asgi_app(scope, receive, send) -> None:
    """수신 본문 길이를 그대로 돌려주는 최소 ASGI 앱."""
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            more_body = False
            continue
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(len(body)).encode()})


async def _drive(middleware, frames: list[dict], headers=None) -> list[dict]:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers or [],
    }
    iterator = iter(frames)

    async def receive() -> dict:
        try:
            return next(iterator)
        except StopIteration:
            return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


async def test_rejects_multiframe_body_over_limit_without_content_length():
    """Content-Length가 없어도 실제 누적 바이트가 상한을 넘으면 413."""
    middleware = BodySizeLimitMiddleware(_echo_asgi_app, max_body_bytes=50)
    frames = [
        {"type": "http.request", "body": b"x" * 30, "more_body": True},
        {"type": "http.request", "body": b"x" * 30, "more_body": False},
    ]

    sent = await _drive(middleware, frames)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


async def test_allows_multiframe_body_within_limit():
    """다중 프레임이라도 상한 이내면 전체 본문이 앱에 전달된다."""
    middleware = BodySizeLimitMiddleware(_echo_asgi_app, max_body_bytes=50)
    frames = [
        {"type": "http.request", "body": b"x" * 10, "more_body": True},
        {"type": "http.request", "body": b"x" * 10, "more_body": False},
    ]

    sent = await _drive(middleware, frames)

    start = next(m for m in sent if m["type"] == "http.response.start")
    body = next(m for m in sent if m["type"] == "http.response.body")
    assert start["status"] == 200
    assert body["body"] == b"20"
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
