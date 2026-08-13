from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.infrastructure.virustotal.client import VirusTotalClient


def _vt_report_response(stats: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": {"attributes": {"last_analysis_stats": stats}}},
        request=httpx.Request("GET", "https://www.virustotal.com/api/v3/urls/dummy"),
    )


@pytest.mark.asyncio
async def test_raw_score_is_ratio_of_malicious_engines_to_total():
    """
    악성 판정 엔진 수를 전체 엔진 수로 나눈 '비율'로 raw_score가 산정되어야 한다
    (기존엔 malicious*0.15 같은 임의 개수 기반 공식이었음).
    """
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    stats = {
        "malicious": 9,
        "suspicious": 0,
        "harmless": 81,
        "undetected": 0,
        "timeout": 0,
    }  # 총 90개 중 9개
    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response(stats),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["total_engines"] == 90
    assert result["raw_score"] == round(9 / 90, 2)


@pytest.mark.asyncio
async def test_raw_score_includes_half_weighted_suspicious_ratio():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    stats = {
        "malicious": 0,
        "suspicious": 10,
        "harmless": 90,
        "undetected": 0,
        "timeout": 0,
    }  # 총 100개 중 10개 의심
    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response(stats),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["raw_score"] == round((10 / 100) * 0.5, 2)


@pytest.mark.asyncio
async def test_raw_score_same_detection_count_scores_lower_with_larger_engine_pool():
    """
    같은 탐지 엔진 수라도 전체 엔진 풀이 클수록(비율이 낮아지므로) raw_score가 낮게 나와야
    비율 기반 산정이 실제로 반영된 것이다.
    """
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    small_pool_stats = {"malicious": 5, "harmless": 15}  # 20개 중 5개 = 25%
    large_pool_stats = {"malicious": 5, "harmless": 95}  # 100개 중 5개 = 5%

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response(small_pool_stats),
    ):
        small_pool_result = await engine.scan_url("https://example.com")

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response(large_pool_stats),
    ):
        large_pool_result = await engine.scan_url("https://example.com")

    assert small_pool_result["raw_score"] > large_pool_result["raw_score"]


@pytest.mark.asyncio
async def test_raw_score_is_zero_when_no_engines_reported():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response({}),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["raw_score"] == 0.0
    assert result["total_engines"] == 0


@pytest.mark.asyncio
async def test_missing_api_key_returns_unavailable_without_calling_api():
    engine = VirusTotalClient()
    engine.api_key = None

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as get:
        result = await engine.scan_url("https://example.com")

    get.assert_not_called()
    assert result["status"] == "unavailable"
    assert result["error_code"] == "MISSING_API_KEY"


@pytest.mark.asyncio
async def test_rate_limited_429_on_report_returns_unavailable_rate_limited():
    """VT는 raise_for_status() 전에 429를 직접 검사하는 별도 분기가 있어
    HTTPStatusError 경로와는 별개로 검증이 필요하다."""
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            429,
            request=httpx.Request(
                "GET", "https://www.virustotal.com/api/v3/urls/dummy"
            ),
        ),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "RATE_LIMITED"
    assert result["is_malicious"] is False


@pytest.mark.asyncio
async def test_server_error_returns_unavailable_with_http_status_code():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=httpx.Response(
            503,
            request=httpx.Request(
                "GET", "https://www.virustotal.com/api/v3/urls/dummy"
            ),
        ),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "HTTP_503"


@pytest.mark.asyncio
async def test_timeout_returns_unavailable_timeout():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("timed out"),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_network_error_returns_unavailable_network_error():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "NETWORK_ERROR"


@pytest.mark.asyncio
async def test_unexpected_exception_does_not_propagate():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "UNEXPECTED_ERROR"


@pytest.mark.asyncio
async def test_no_existing_report_requests_new_scan_and_returns_scanning_status():
    """기존 보고서가 없으면(404) 신규 스캔을 요청하고 'scanning' 상태를 반환해야 한다."""
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    get_response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://www.virustotal.com/api/v3/urls/dummy"),
    )
    post_response = httpx.Response(
        200,
        json={"data": {"id": "scan-id"}},
        request=httpx.Request("POST", "https://www.virustotal.com/api/v3/urls"),
    )

    with (
        patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=get_response,
        ),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=post_response,
        ) as post,
    ):
        result = await engine.scan_url("https://never-scanned-before.example")

    post.assert_called_once()
    assert result["status"] == "scanning"
    assert result["is_malicious"] is False
    assert result["error_code"] is None


@pytest.mark.asyncio
async def test_new_scan_request_rate_limited_returns_unavailable():
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    get_response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://www.virustotal.com/api/v3/urls/dummy"),
    )
    post_response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://www.virustotal.com/api/v3/urls"),
    )

    with (
        patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            return_value=get_response,
        ),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=post_response,
        ),
    ):
        result = await engine.scan_url("https://never-scanned-before.example")

    assert result["status"] == "unavailable"
    assert result["error_code"] == "RATE_LIMITED"


@pytest.mark.asyncio
async def test_is_malicious_threshold_is_unaffected_by_ratio_change():
    """is_malicious 판정(엔진 3개 이상 등)은 비율과 무관하게 기존 절대 개수 기준을 그대로 유지해야 한다."""
    engine = VirusTotalClient()
    engine.api_key = "dummy-key"

    stats = {
        "malicious": 3,
        "harmless": 87,
    }  # 90개 중 3개 (비율은 낮지만 절대개수 3개는 여전히 malicious)
    with patch.object(
        httpx.AsyncClient,
        "get",
        new_callable=AsyncMock,
        return_value=_vt_report_response(stats),
    ):
        result = await engine.scan_url("https://example.com")

    assert result["is_malicious"] is True
