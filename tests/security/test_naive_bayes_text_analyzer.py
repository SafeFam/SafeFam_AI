import pytest
from pathlib import Path
from app.service.security import naive_bayes_text_analyzer as nb


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """모듈 전역 캐시가 테스트 간에 새서 로드 성공/실패 케이스가 서로 오염되지 않도록 초기화."""
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


def test_normalize_text_masks_url_and_amount():
    text = "http://bit.ly/fake 계좌로 500,000원 즉시 입금하세요"
    normalized = nb._normalize_text(text)

    assert "<URL>" in normalized
    assert "<금액>" in normalized
    assert "http://" not in normalized


def test_extract_struct_features_detects_short_url_and_phone():
    text = "010-1234-5678 로 연락주세요 http://bit.ly/fake"
    features = nb._extract_struct_features(text)

    # [has_url, has_short_url, has_phone, has_amount, has_web_tag, is_long_text]
    assert features[0] == 1  # has_url
    assert features[1] == 1  # has_short_url
    assert features[2] == 1  # has_phone
    assert features[3] == 0  # has_amount
    assert features[4] == 0  # has_web_tag
    assert features[5] == 0  # is_long_text


@pytest.mark.asyncio
async def test_analyze_text_with_naive_bayes_missing_artifact_is_fail_safe(monkeypatch):
    """
    모델 아티팩트를 찾을 수 없을 때 예외를 던지는 대신 UNKNOWN 등급 + 에러 메시지로
    안전하게 대체되는지 검증합니다 (fail-safe fallback).
    """
    monkeypatch.setattr(nb, "MODEL_PATH", Path("/nonexistent/model.pkl"))
    monkeypatch.setattr(nb, "VECTORIZER_PATH", Path("/nonexistent/vectorizer.pkl"))

    result = await nb.analyze_text_with_naive_bayes("테스트 메시지")

    assert result["is_available"] is False
    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["error_message"] is not None


@pytest.mark.asyncio
async def test_analyze_text_with_naive_bayes_real_model_flags_phishing():
    """
    저장소에 실제로 커밋된 사전 학습 아티팩트를 로드하여, 전형적인 스미싱 문장이
    높은 위험 점수로 판정되는지 통합 검증합니다.
    """
    text = "[국민건강보험] 건강검진 보고서 발급 완료. 즉시 확인하세요 http://bit.ly/fake"

    result = await nb.analyze_text_with_naive_bayes(text)

    assert result["is_available"] is True
    assert result["result"]["risk_score"] > 50
    assert result["result"]["grade"] in ("SUSPICIOUS", "DANGEROUS")


@pytest.mark.asyncio
async def test_analyze_text_with_naive_bayes_real_model_flags_casual_text_as_safe():
    text = "오늘 소주 한잔 고?"

    result = await nb.analyze_text_with_naive_bayes(text)

    assert result["is_available"] is True
    assert result["result"]["grade"] == "SAFE"
