from unittest.mock import AsyncMock

import pytest

from app.analysis.url.analyzer import HybridUrlAnalyzer


@pytest.mark.asyncio
async def test_gsb_block_marks_is_gsb_confirmed_true_and_skips_vt():
    """
    GSB가 블랙리스트 매치를 확인하면 is_gsb_confirmed=True로 표시되어야 하고,
    VT는 quota 절약을 위해 호출되지 않아야 한다.
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "raw_score": 0.95,
            "status": "completed",
        }
    )
    engine.vt_client.scan_url = AsyncMock()

    result = await engine.scan_url("https://danger-phishing-test-site.com")

    assert result["is_malicious"] is True
    assert result["is_gsb_confirmed"] is True
    engine.vt_client.scan_url.assert_not_called()


@pytest.mark.asyncio
async def test_vt_only_detection_does_not_mark_gsb_confirmed():
    """
    GSB는 깨끗한데 VT 백업에서 악성이 잡힌 경우, is_malicious=True는 될 수 있어도
    is_gsb_confirmed는 False여야 한다 (GSB 확정 오버라이드 트리거 대상이 아님).
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "detected_count": 5,
            "raw_score": 0.8,
            "status": "completed",
        }
    )

    result = await engine.scan_url("https://hidden-malware-link.xyz")

    assert result["is_malicious"] is True
    assert result["is_gsb_confirmed"] is False


@pytest.mark.asyncio
async def test_vt_weak_detection_below_threshold_is_not_confirmed():
    """
    VT 탐지 엔진 수가 임계치(5) 미만인 소수 탐지는 확정 취급하지 않고
    기존처럼 가중치 기반 점수로만 반영되어야 한다 (오탐 벤더 섞일 가능성 고려).
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "detected_count": 2,
            "raw_score": 0.3,
            "status": "completed",
        }
    )

    result = await engine.scan_url("https://borderline-site.example")

    assert result["is_malicious"] is True
    assert result["is_vt_confirmed"] is False


@pytest.mark.asyncio
async def test_vt_strong_consensus_at_or_above_threshold_is_confirmed():
    """
    VT 탐지 엔진 수가 임계치(5) 이상이면 다수 백신사 합의로 보고 확정 악성으로 승격되어야 한다.
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "detected_count": 7,
            "raw_score": 0.9,
            "status": "completed",
        }
    )

    result = await engine.scan_url("https://hidden-malware-link.xyz")

    assert result["is_vt_confirmed"] is True


@pytest.mark.asyncio
async def test_gsb_blocked_branch_skips_vt_so_vt_confirmed_is_false():
    """GSB가 이미 차단해서 VT 호출 자체를 생략한 경우 is_vt_confirmed는 False여야 한다."""
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "raw_score": 0.95,
            "status": "completed",
        }
    )
    engine.vt_client.scan_url = AsyncMock()

    result = await engine.scan_url("https://danger-phishing-test-site.com")

    assert result["is_vt_confirmed"] is False


@pytest.mark.asyncio
async def test_clean_url_is_not_gsb_confirmed():
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "detected_count": 0,
            "status": "safe",
        }
    )

    result = await engine.scan_url("https://www.google.com")

    assert result["is_malicious"] is False
    assert result["is_gsb_confirmed"] is False


@pytest.mark.asyncio
async def test_gsb_unavailable_falls_back_to_vt_result():
    """GSB 장애 시 VT 결과만으로도 트랙이 계속 동작해야 한다(전체 중단 X)."""
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "unavailable",
            "error_code": "TIMEOUT",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": True,
            "detected_count": 7,
            "raw_score": 0.9,
            "status": "completed",
        }
    )

    result = await engine.scan_url("https://example.com")

    assert result["available"] is True
    assert result["is_malicious"] is True
    assert result["failed_providers"] == ["GSB"]
    assert result["provider_error_codes"] == {"GSB": "TIMEOUT"}
    engine.vt_client.scan_url.assert_called_once()


@pytest.mark.asyncio
async def test_vt_unavailable_after_clean_gsb_still_marks_track_available():
    """GSB가 정상 응답(안전)했다면 VT가 죽어도 트랙 자체는 available로 유지돼야 한다."""
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "unavailable",
            "error_code": "RATE_LIMITED",
        }
    )

    result = await engine.scan_url("https://example.com")

    assert result["available"] is True
    assert result["is_malicious"] is False
    assert result["failed_providers"] == ["VIRUSTOTAL"]
    assert result["provider_error_codes"] == {"VIRUSTOTAL": "RATE_LIMITED"}


@pytest.mark.asyncio
async def test_both_providers_unavailable_marks_track_unavailable_not_malicious():
    """
    GSB, VT 둘 다 죽었을 때 URL 트랙 전체가 unavailable로 표시돼야 하며,
    이때 is_malicious를 True로 fail-open 시키지 않는다(점수 재분배는 scoring 레이어 책임).
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "unavailable",
            "error_code": "TIMEOUT",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "unavailable",
            "error_code": "NETWORK_ERROR",
        }
    )

    result = await engine.scan_url("https://example.com")

    assert result["available"] is False
    assert result["is_malicious"] is False
    assert set(result["failed_providers"]) == {"GSB", "VIRUSTOTAL"}
    assert result["error_message"] is not None


@pytest.mark.asyncio
async def test_gsb_client_raising_exception_is_absorbed_as_unavailable():
    """클라이언트가 unavailable dict가 아니라 예외 자체를 던져도 파이프라인이 죽지 않아야 한다."""
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(side_effect=RuntimeError("boom"))
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )

    result = await engine.scan_url("https://example.com")

    assert result["failed_providers"] == ["GSB"]
    assert result["provider_error_codes"]["GSB"] == "UNEXPECTED_ERROR"
    assert result["available"] is True


@pytest.mark.asyncio
async def test_vt_client_raising_exception_is_absorbed_as_unavailable():
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(side_effect=RuntimeError("boom"))

    result = await engine.scan_url("https://example.com")

    assert result["failed_providers"] == ["VIRUSTOTAL"]
    assert result["provider_error_codes"]["VIRUSTOTAL"] == "UNEXPECTED_ERROR"
    assert result["available"] is True


@pytest.mark.asyncio
async def test_single_vt_detection_is_not_malicious():
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "status": "safe",
        }
    )
    engine.vt_client.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "detected_count": 1,
            "raw_score": 0.1,
            "status": "safe",
        }
    )

    result = await engine.scan_url("https://example.com")

    assert result["is_malicious"] is False
