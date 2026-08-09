import asyncio
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

from app.core.config import settings
from app.infrastructure.http_retry import request_with_retry

logger = logging.getLogger(__name__)

# URL 정규표현식 패턴
URL_PATTERN = re.compile(r'https?://[^\s\'"<>]+')


def _is_blocked_ip(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """사설/루프백/링크로컬 등 외부에 공개되지 않은 주소인지 판별."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def _is_public_host(
    hostname: str | None,
    dns_timeout: float | None = None,
) -> str | None:
    """
    SSRF 방어: 호스트가 실제로 가리키는 IP를 DNS로 확인해서 내부망/사설 대역이면 차단.
    도메인이 공개 주소처럼 보여도 리다이렉트 체인 중간에 내부망으로 우회할 수 있으므로
    매 홉마다 이 검증을 거쳐야 함.

    여기서 확인한 IP를 실제 연결에도 그대로 고정(pin)해야 검증-사용 시점 사이에 DNS
    응답이 바뀌는 DNS 리바인딩을 막을 수 있다. 그래서 bool이 아니라 검증에 사용한
    IP 문자열(고정할 주소)을 반환하고, 차단 시 None을 반환한다.
    """
    if dns_timeout is None:
        dns_timeout = settings.URL_TRACE_TIMEOUT_SECONDS

    if not hostname:
        return None

    try:
        ip = ipaddress.ip_address(hostname)
        return None if _is_blocked_ip(ip) else str(ip)
    except ValueError:
        pass  # IP 리터럴이 아니라 도메인 -> DNS 조회 필요

    try:
        loop = asyncio.get_running_loop()
        addr_infos = await asyncio.wait_for(
            loop.getaddrinfo(hostname, None), timeout=dns_timeout
        )
    except (socket.gaierror, OSError, asyncio.TimeoutError):
        logger.warning("[SSRF Guard] DNS 조회 실패 또는 타임아웃으로 요청 차단")
        return None

    pinned_ip = None
    for info in addr_infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return None
        if _is_blocked_ip(ip):
            return None
        if pinned_ip is None:
            pinned_ip = str(ip)

    return pinned_ip


class _PinnedIPBackend(httpcore.AnyIOBackend):
    """
    검증된 (hostname, ip) 쌍에 대해서만 실제 TCP 연결을 강제하는 network backend.
    connect_tcp에 다시 호스트명이 들어와도 자체적으로 DNS를 재조회하지 않고 이미
    검증된 IP로만 연결하므로, 검증 시점과 연결 시점 사이에 DNS 응답이 바뀌는
    DNS 리바인딩 공격을 차단한다. SNI/Host 헤더는 origin 호스트명을 그대로 쓰므로
    TLS 인증서 검증과 가상호스팅에는 영향이 없다.
    """

    def __init__(self, pinned_host: str, pinned_ip: str):
        super().__init__()
        self._pinned_host = pinned_host.lower()
        self._pinned_ip = pinned_ip

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        if host.lower() != self._pinned_host:
            # 이 transport는 검증된 단일 호스트 전용이므로, 다른 호스트로의 연결
            # 요청은 미검증 상태라는 뜻 -> 안전하게 차단 (fail-closed)
            raise httpcore.ConnectError(
                f"[SSRF Guard] 검증되지 않은 호스트로의 연결 시도 차단: {host}"
            )
        return await super().connect_tcp(
            self._pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class _PinnedIPTransport(httpx.AsyncHTTPTransport):
    """
    검증된 IP로 연결을 고정하는 httpx transport. 부모의 __init__을 호출하지 않고
    커스텀 network_backend를 가진 connection pool로 self._pool을 직접 구성한다.
    handle_async_request/aclose는 self._pool만 사용하므로 그대로 상속해 재사용한다.
    """

    def __init__(self, pinned_host: str, pinned_ip: str, verify: bool = True):
        from httpx._config import create_ssl_context

        ssl_context = create_ssl_context(verify=verify, trust_env=True)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            max_connections=10,
            max_keepalive_connections=5,
            network_backend=_PinnedIPBackend(pinned_host, pinned_ip),
        )


# 텍스트에서 URL를 추출하고 정제, 중복 제거하여 반환
def extract_urls(text: str) -> list[str]:

    if not text:
        return []

    raw_urls = URL_PATTERN.findall(text)
    cleaned_urls = []

    for url in raw_urls:
        cleaned_url = url.rstrip(".,?!:;)[]")
        cleaned_urls.append(cleaned_url)

    return list(dict.fromkeys(cleaned_urls))


# 단축 URL의 리다이렉트를 추적하고 최종 주소를 반환
async def trace_url(
    url: str,
    max_redirects: int = 5,
    timeout: float | None = None,
) -> str:
    if timeout is None:
        timeout = settings.URL_TRACE_TIMEOUT_SECONDS

    current_url = url

    for attempt in range(max_redirects):
        try:
            hostname = urlparse(current_url).hostname
            pinned_ip = await _is_public_host(hostname, dns_timeout=timeout)
            if not pinned_ip:
                logger.warning(
                    "[SSRF Guard] 내부망 또는 사설 주소로 판단되어 요청 차단"
                )
                break

            # 검증에 쓴 IP를 그대로 연결에 고정(pin)한다. 홉마다 호스트가 바뀔 수 있으므로
            # 매 홉마다 그 홉 전용 transport를 새로 만들어, 검증-연결 사이에 DNS가
            # 바뀌어도(리바인딩) 다른 주소로 새지 않게 한다.
            transport = _PinnedIPTransport(pinned_host=hostname, pinned_ip=pinned_ip)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

            async with httpx.AsyncClient(
                transport=transport, limits=limits, follow_redirects=False
            ) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }

                response = await request_with_retry(
                    lambda current_url=current_url, headers=headers: client.head(
                        current_url,
                        headers=headers,
                        timeout=timeout,
                    ),
                    max_retries=settings.EXTERNAL_API_MAX_RETRIES,
                    operation_name="URL trace HEAD",
                )

                # HEAD를 차단하거나 거부하는 서버(400, 404, 405)에 대응하기 위한 GET 폴백
                if response.status_code in [400, 404, 405]:
                    response = await request_with_retry(
                        lambda current_url=current_url, headers=headers: client.get(
                            current_url,
                            headers=headers,
                            timeout=timeout,
                        ),
                        max_retries=settings.EXTERNAL_API_MAX_RETRIES,
                        operation_name="URL trace GET",
                    )

            # HTTP Redirection 상태 코드 판별 (3xx)
            if response.is_redirect or response.status_code in [
                301,
                302,
                303,
                307,
                308,
            ]:
                location = response.headers.get("Location")
                if not location:
                    break

                # 절대 URL이 아닌 모든 경우(경로만/프로토콜 상대경로 등)를 urljoin으로 정규화
                if not location.startswith(("http://", "https://")):
                    location = urljoin(current_url, location)

                current_url = location
                logger.info(
                    "URL redirect followed. redirect_count=%d",
                    attempt + 1,
                )
            else:
                break

        except httpx.TimeoutException:
            logger.warning(
                "URL 추적 타임아웃 발생. timeout_seconds=%s",
                timeout,
            )
            break
        except httpx.HTTPStatusError as exception:
            logger.error(
                "URL 추적 HTTP 오류. status_code=%s",
                exception.response.status_code,
            )
            break
        except Exception as exception:
            logger.error(
                "URL 추적 실패. error_type=%s",
                type(exception).__name__,
            )
            break
    else:
        logger.warning(
            f"최대 리다이렉트 횟수({max_redirects}회)를 초과했습니다. 루프 위험 감지."
        )

    return current_url
