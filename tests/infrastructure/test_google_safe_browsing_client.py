from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.infrastructure.google_safe_browsing.client import (
    GoogleSafeBrowsingClient,
)


def _gsb_response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=json_body or {},
        request=httpx.Request(
            "POST", "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        ),
    )


@pytest.mark.asyncio
async def test_missing_api_key_returns_unavailable_without_calling_api():
    client = GoogleSafeBrowsingClient()
    client.api_key = None

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as post:
        result = await client.scan_url("https://example.com")

    post.assert_not_called()
    assert result["status"] == "unavailable"
    assert result["error_code"] == "MISSING_API_KEY"
    assert result["is_malicious"] is False


@pytest.mark.asyncio
async def test_rate_limited_429_returns_unavailable_rate_limited():
    """GSB는 VT와 달리 상태 코드를 먼저 검사하지 않고 raise_for_status()로 넘기므로,
    429 -> HTTPStatusError -> RATE_LIMITED 매핑 경로를 별도로 검증해야 한다."""
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        return_value=_gsb_response(429),
    ):
        result = await client.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "RATE_LIMITED"
    assert result["is_malicious"] is False


@pytest.mark.asyncio
async def test_server_error_returns_unavailable_with_http_status_code():
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        return_value=_gsb_response(503),
    ):
        result = await client.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "HTTP_503"


@pytest.mark.asyncio
async def test_timeout_returns_unavailable_timeout():
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("timed out"),
    ):
        result = await client.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_network_error_returns_unavailable_network_error():
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = await client.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "NETWORK_ERROR"


@pytest.mark.asyncio
async def test_unexpected_exception_does_not_propagate():
    """예측 못 한 예외까지도 파이프라인을 죽이지 않고 unavailable로 흡수해야 한다."""
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await client.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "UNEXPECTED_ERROR"


@pytest.mark.asyncio
async def test_threat_match_returns_malicious_result():
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        return_value=_gsb_response(
            200,
            {"matches": [{"threatType": "SOCIAL_ENGINEERING"}]},
        ),
    ):
        result = await client.scan_url("https://phishing.example")

    assert result["is_malicious"] is True
    assert result["status"] == "completed"
    assert result["detected_count"] == 1


@pytest.mark.asyncio
async def test_no_match_returns_safe_result():
    client = GoogleSafeBrowsingClient()
    client.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        return_value=_gsb_response(200, {}),
    ):
        result = await client.scan_url("https://example.com")

    assert result["is_malicious"] is False
    assert result["status"] == "safe"
    assert result["error_code"] is None
