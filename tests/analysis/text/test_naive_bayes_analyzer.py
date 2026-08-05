"""Naive Bayes API 분석기의 모델 로딩 및 응답 회귀 테스트."""

from pathlib import Path

import pytest

from app.analysis.text import naive_bayes_analyzer as nb


@pytest.fixture(autouse=True)
def reset_model_cache():
    """모델 전역 캐시가 테스트 사이에 공유되지 않도록 초기화합니다."""
    nb._model = None
    nb._vectorizer = None
    nb._threshold = None
    nb._classes = None
    nb._load_error = None
    nb._load_attempted = False

    yield

    nb._model = None
    nb._vectorizer = None
    nb._threshold = None
    nb._classes = None
    nb._load_error = None
    nb._load_attempted = False


@pytest.mark.asyncio
async def test_missing_artifact_returns_fail_safe_result(
    monkeypatch,
    tmp_path: Path,
):
    """아티팩트가 없을 때 예외 대신 UNKNOWN 결과를 반환합니다."""
    monkeypatch.setattr(nb, "MODEL_PATH", tmp_path / "missing-model.pkl")
    monkeypatch.setattr(
        nb,
        "VECTORIZER_PATH",
        tmp_path / "missing-vectorizer.pkl",
    )

    result = await nb.analyze_text_with_naive_bayes("테스트 메시지")

    assert result["engine"] == "naive_bayes"
    assert result["is_available"] is False
    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["risk_score"] == 0
    assert result["result"]["is_suspected_phishing"] is False
    assert result["result"]["error_message"] is not None


@pytest.mark.asyncio
async def test_real_model_flags_phishing():
    """기존 모델이 전형적인 피싱 문장을 계속 탐지하는지 확인합니다."""
    text = (
        "[국민건강보험] 건강검진 보고서 발급 완료. "
        "즉시 확인하세요 http://bit.ly/fake"
    )

    result = await nb.analyze_text_with_naive_bayes(text)

    assert result["is_available"] is True
    assert result["result"]["risk_score"] > 50
    assert result["result"]["grade"] in ("SUSPICIOUS", "DANGEROUS")
    assert result["result"]["is_suspected_phishing"] is True
    assert result["result"]["error_message"] is None


@pytest.mark.asyncio
async def test_real_model_flags_casual_text_as_safe():
    """기존 모델이 일반 대화를 계속 안전으로 판단하는지 확인합니다."""
    result = await nb.analyze_text_with_naive_bayes("오늘 소주 한잔 고?")

    assert result["is_available"] is True
    assert result["result"]["grade"] == "SAFE"
    assert result["result"]["is_suspected_phishing"] is False
    assert result["result"]["error_message"] is None
