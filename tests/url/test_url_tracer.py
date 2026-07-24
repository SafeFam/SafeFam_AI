import pytest
import httpx
from unittest.mock import patch, AsyncMock
from app.utils.url_tracker import trace_url


@pytest.mark.asyncio
async def test_normal_url_no_redirect():
    """일반 URL은 리다이렉트 없이 자기 자신을 반환해야 함"""
    url = "https://www.google.com"
    result = await trace_url(url)
    assert result == url


@pytest.mark.asyncio
async def test_tinyurl_resolution():
    """tinyurl 단축 주소가 원본 구글 주소로 잘 풀리는지 검증"""
    short_url = "https://tinyurl.com/app-store"
    result = await trace_url(short_url)
    # 종착지가 특정 앱스토어 링크나 정상 도메인으로 복원되는지 확인
    assert "tinyurl.com" not in result


@pytest.mark.asyncio
async def test_broken_url_graceful_handling():
    """존재하지 않는 이상한 URL이어도 서버가 안 터지고 본래 값을 뱉어내는지 검증"""
    invalid_url = "https://this-is-completely-broken-domain-12345.com"
    result = await trace_url(invalid_url, timeout=2.0)
    assert result == invalid_url


@pytest.mark.asyncio
async def test_max_redirect_limit():
    """최대 리다이렉트 제한 횟수가 정상 작동하는지 (0회 제한 테스트)"""
    short_url = "https://tinyurl.com/app-store"
    # max_redirects를 0으로 주면 바로 튕겨서 원래 주소가 나와야 함
    result = await trace_url(short_url, max_redirects=0)
    assert result == short_url


@pytest.mark.asyncio
async def test_relative_redirect_without_leading_slash_is_resolved():
    """
    Location 헤더가 절대 URL이 아니고 '/'로 시작하지도 않는 상대경로(예: 'next-page')를 줄 때도
    urljoin으로 현재 경로 기준 정규화되어야 한다 (과거엔 '/'로 시작하는 경우만 처리해서
    이런 케이스는 깨진 상대경로를 그대로 요청해버리는 버그가 있었음).
    """
    redirect_response = httpx.Response(
        301,
        headers={"Location": "next-page"},
        request=httpx.Request("HEAD", "https://short.example/a/b"),
    )
    final_response = httpx.Response(
        200,
        request=httpx.Request("HEAD", "https://short.example/a/next-page"),
    )

    with patch.object(httpx.AsyncClient, "head", new_callable=AsyncMock) as mock_head, \
         patch("app.utils.url_tracker._is_public_host", new_callable=AsyncMock, return_value=True):
        mock_head.side_effect = [redirect_response, final_response]
        result = await trace_url("https://short.example/a/b")

    assert result == "https://short.example/a/next-page"


@pytest.mark.asyncio
async def test_protocol_relative_redirect_is_resolved():
    """Location 헤더가 '//example.com/path' 형태(프로토콜 상대경로)인 경우도 정규화되어야 한다."""
    redirect_response = httpx.Response(
        302,
        headers={"Location": "//other.example/landing"},
        request=httpx.Request("HEAD", "https://short.example/a"),
    )
    final_response = httpx.Response(
        200,
        request=httpx.Request("HEAD", "https://other.example/landing"),
    )

    with patch.object(httpx.AsyncClient, "head", new_callable=AsyncMock) as mock_head, \
         patch("app.utils.url_tracker._is_public_host", new_callable=AsyncMock, return_value=True):
        mock_head.side_effect = [redirect_response, final_response]
        result = await trace_url("https://short.example/a")

    assert result == "https://other.example/landing"


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_loopback_ip_literal():
    """루프백 IP 리터럴로의 요청은 DNS 조회 없이도 즉시 차단되어야 한다 (SSRF 방지)."""
    result = await trace_url("http://127.0.0.1:8080/admin")
    assert result == "http://127.0.0.1:8080/admin"


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_cloud_metadata_ip():
    """클라우드 인스턴스 메타데이터 주소(169.254.169.254)로의 요청은 차단되어야 한다."""
    result = await trace_url("http://169.254.169.254/latest/meta-data/")
    assert result == "http://169.254.169.254/latest/meta-data/"


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_private_lan_ip():
    """사설 대역(예: 192.168.0.0/16)으로의 요청은 차단되어야 한다."""
    result = await trace_url("http://192.168.1.1/")
    assert result == "http://192.168.1.1/"


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_redirect_that_pivots_to_internal_ip():
    """
    첫 요청은 공개 도메인이라 통과하더라도, 리다이렉트 체인 중간에
    내부망 주소로 튀는 경우 그 지점에서 차단하고 마지막으로 안전했던 URL을 반환해야 한다.
    """
    redirect_response = httpx.Response(
        302,
        headers={"Location": "http://127.0.0.1/internal-admin"},
        request=httpx.Request("HEAD", "https://short.example/a"),
    )

    with patch.object(httpx.AsyncClient, "head", new_callable=AsyncMock) as mock_head, \
         patch("app.utils.url_tracker._is_public_host", new_callable=AsyncMock, side_effect=[True, False]):
        mock_head.side_effect = [redirect_response]
        result = await trace_url("https://short.example/a")

    # 내부망 URL로는 실제 요청이 나가지 않고, 리다이렉트 직전 URL에서 멈춰야 한다
    assert result == "http://127.0.0.1/internal-admin"
    assert mock_head.call_count == 1
