import logging
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# 재시도 대상이 되는 httpx 네트워크 예외 목록
_RETRYABLE_REQUEST_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def is_retryable_status(status_code: int) -> bool:
    """HTTP 상태 코드가 재시도 가능한 대상인지 검증"""
    return (
        status_code == 429
        or 500 <= status_code <= 599
    )


async def request_with_retry(
    operation: Callable[[], Awaitable[httpx.Response]],
    *,
    max_retries: int,
    operation_name: str,
) -> httpx.Response:
    """비동기 HTTP 요청 중 네트워크 예외 또는 재시도 대상 상태 코드가 발생하면 지정된 횟수만큼 재시도 수행"""
    for attempt in range(max_retries + 1):
        try:
            response = await operation()

        except _RETRYABLE_REQUEST_ERRORS as exc:
            # 최대 재시도 횟수 도달 시 예외 전차
            if attempt >= max_retries:
                raise

            logger.warning(
                "%s 호출 실패로 재시도합니다. attempt=%s/%s error=%s",
                operation_name,
                attempt + 1,
                max_retries,
                type(exc).__name__,
            )
            continue

        # 기존 커넥션 정리 후 재시도
        if (
            is_retryable_status(response.status_code)
            and attempt < max_retries
        ):
            logger.warning(
                "%s 응답이 재시도 대상입니다. "
                "attempt=%s/%s status=%s",
                operation_name,
                attempt + 1,
                max_retries,
                response.status_code,
            )
            await response.aclose()
            continue

        return response

    raise RuntimeError(
        f"{operation_name} retry loop terminated unexpectedly"
    )