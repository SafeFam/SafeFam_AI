from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """Content-Length가 상한을 초과하는 요청을 파싱 전에 413으로 거부한다(issue #120).

    스키마 레벨 문자 수 제한이 1차 방어이며, 이 미들웨어는 초대형 바디가 메모리에
    적재/역직렬화되기 전에 차단하는 방어 심화(defense-in-depth) 계층이다.
    정상 JSON 요청(BE의 httpx, 테스트 클라이언트 등)은 Content-Length를 항상 보낸다.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if (
            content_length is not None
            and content_length.isdigit()
            and int(content_length) > self.max_body_bytes
        ):
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
