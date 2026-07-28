import pytest
from unittest.mock import AsyncMock
from app.analysis.url.analyzer import HybridUrlAnalyzer


@pytest.mark.asyncio
async def test_gsb_block_marks_is_gsb_confirmed_true_and_skips_vt():
    """
    GSB가 블랙리스트 매치를 확인하면 is_gsb_confirmed=True로 표시되어야 하고,
    VT는 quota 절약을 위해 호출되지 않아야 한다.
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": True,
        "raw_score": 0.95,
        "status": "completed",
    })
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
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "status": "safe",
    })
    engine.vt_client.scan_url = AsyncMock(return_value={
        "is_malicious": True,
        "detected_count": 5,
        "raw_score": 0.8,
        "status": "completed",
    })

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
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "status": "safe",
    })
    engine.vt_client.scan_url = AsyncMock(return_value={
        "is_malicious": True,
        "detected_count": 2,
        "raw_score": 0.3,
        "status": "completed",
    })

    result = await engine.scan_url("https://borderline-site.example")

    assert result["is_malicious"] is True
    assert result["is_vt_confirmed"] is False


@pytest.mark.asyncio
async def test_vt_strong_consensus_at_or_above_threshold_is_confirmed():
    """
    VT 탐지 엔진 수가 임계치(5) 이상이면 다수 백신사 합의로 보고 확정 악성으로 승격되어야 한다.
    """
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "status": "safe",
    })
    engine.vt_client.scan_url = AsyncMock(return_value={
        "is_malicious": True,
        "detected_count": 7,
        "raw_score": 0.9,
        "status": "completed",
    })

    result = await engine.scan_url("https://hidden-malware-link.xyz")

    assert result["is_vt_confirmed"] is True


@pytest.mark.asyncio
async def test_gsb_blocked_branch_skips_vt_so_vt_confirmed_is_false():
    """GSB가 이미 차단해서 VT 호출 자체를 생략한 경우 is_vt_confirmed는 False여야 한다."""
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": True,
        "raw_score": 0.95,
        "status": "completed",
    })
    engine.vt_client.scan_url = AsyncMock()

    result = await engine.scan_url("https://danger-phishing-test-site.com")

    assert result["is_vt_confirmed"] is False


@pytest.mark.asyncio
async def test_clean_url_is_not_gsb_confirmed():
    engine = HybridUrlAnalyzer()
    engine.gsb_client.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "status": "safe",
    })
    engine.vt_client.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "detected_count": 0,
        "status": "safe",
    })

    result = await engine.scan_url("https://www.google.com")

    assert result["is_malicious"] is False
    assert result["is_gsb_confirmed"] is False
