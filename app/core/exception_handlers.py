from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 422 Unprocessable Content. 상수명이 starlette 버전에 따라
# (ENTITY/CONTENT) 달라 deprecation을 피하려 리터럴을 사용한다.
_HTTP_422 = 422


def _sanitize_errors(errors: list[dict]) -> list[dict]:
    """검증 에러에서 원문(PII 가능) `input`과 직렬화 불가한 `ctx`를 제거한다.

    `type`/`loc`/`msg`만 남겨 클라이언트가 어떤 필드가 왜 거부됐는지는 알되,
    보낸 원문이 응답에 되돌아오지 않도록 한다.
    """
    return [
        {
            "type": error.get("type"),
            "loc": error.get("loc"),
            "msg": error.get("msg"),
        }
        for error in errors
    ]


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """요청 본문을 되비추지 않는 422 응답을 반환한다.

    FastAPI 기본 핸들러는 `exc.errors()`를 그대로 직렬화하는데, 이 목록은
    `hide_input_in_errors`를 켜도 원문 `input`을 포함한다. 안티스미싱 서비스에서
    입력은 문자 본문(PII)이므로 응답에 절대 반사되면 안 된다.
    """
    return JSONResponse(
        status_code=_HTTP_422,
        content={"detail": _sanitize_errors(exc.errors())},
    )
