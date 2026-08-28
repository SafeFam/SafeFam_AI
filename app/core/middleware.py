from starlette.datastructures import Headers
from starlette.responses import JSONResponse
<<<<<<< HEAD
from starlette.types import ASGIApp, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """Content-Length가 상한을 초과하는 요청을 파싱 전에 413으로 거부한다(issue #120).

    스키마 레벨 문자 수 제한이 1차 방어이며, 이 미들웨어는 초대형 바디가 메모리에
    적재/역직렬화되기 전에 차단하는 방어 심화(defense-in-depth) 계층이다.
    정상 JSON 요청(BE의 httpx, 테스트 클라이언트 등)은 Content-Length를 항상 보낸다.
=======
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """요청 본문이 상한을 초과하면 파싱 전에 413으로 거부한다(issue #120).

    스키마 레벨 문자 수 제한이 1차 방어이며, 이 미들웨어는 초대형 바디가
    역직렬화되기 전에 차단하는 방어 심화(defense-in-depth) 계층이다.
    Content-Length 헤더에만 의존하지 않는다 — 헤더가 없거나 위조된(청크 전송 등)
    경우에도 ASGI `http.request` 프레임을 상한까지만 버퍼링해 초과 시 거부하므로
    우회할 수 없다. 메모리 사용량은 `max_body_bytes`로 제한된다.
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
<<<<<<< HEAD
=======
        """HTTP 요청 본문 크기를 강제하고, 이내면 버퍼를 앱에 재생(replay)한다."""
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

<<<<<<< HEAD
=======
        # 빠른 경로: 선언된 Content-Length가 이미 상한을 넘으면 즉시 거부.
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        content_length = Headers(scope=scope).get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
<<<<<<< HEAD
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
=======
            await self._reject(scope, receive, send)
            return

        # 헤더와 무관하게 실제 바이트를 상한까지만 누적한다.
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                await self._replay(scope, send, bytes(body), disconnected=True)
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        await self._replay(scope, send, bytes(body))

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        """413 Payload Too Large 응답을 전송한다(원문 미포함)."""
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"},
        )
        await response(scope, receive, send)

    async def _replay(
        self,
        scope: Scope,
        send: Send,
        body: bytes,
        *,
        disconnected: bool = False,
    ) -> None:
        """버퍼링한 본문을 단일 프레임으로 앱에 재생한다."""
        if disconnected:
            messages: list[Message] = [{"type": "http.disconnect"}]
        else:
            messages = [
                {"type": "http.request", "body": body, "more_body": False}
            ]
        iterator = iter(messages)

        async def replay_receive() -> Message:
            try:
                return next(iterator)
            except StopIteration:
                return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
