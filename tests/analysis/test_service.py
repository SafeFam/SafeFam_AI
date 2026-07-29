import pytest
from unittest.mock import patch, AsyncMock
from app.analysis.service import SmishingAnalysisService


def _nb_result(grade: str, risk_score: int, is_available: bool = True) -> dict:
    return {
        "engine": "naive_bayes",
        "is_available": is_available,
        "result": {
            "grade": grade,
            "risk_score": risk_score,
            "is_suspected_phishing": grade != "SAFE",
            "error_message": None if is_available else "Model Load Error"
        }
    }


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_gemini", new_callable=AsyncMock)
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_hybrid_text_track_skips_gemini_when_naive_bayes_is_safe(mock_nb, mock_gemini):
    """
    1차 나이브 베이즈가 SAFE로 판정하면 Gemini API를 호출하지 않고
    나이브 베이즈 결과를 그대로 text_analysis로 사용해야 한다.
    Gemini가 안 돌았으므로 llm_available은 False, naive_bayes_score는 그 점수를 그대로 반환.
    """
    mock_nb.return_value = _nb_result("SAFE", 12)

    service = SmishingAnalysisService()
    text_analysis, naive_bayes_score, llm_available = await service._analyze_text_hybrid("엄마 오늘 저녁 메뉴 뭐야?")

    mock_gemini.assert_not_called()
    assert text_analysis["engine"] == "naive_bayes"
    assert text_analysis["result"]["risk_score"] == 12
    assert naive_bayes_score == 12
    assert llm_available is False


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_gemini", new_callable=AsyncMock)
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_hybrid_text_track_escalates_to_gemini_when_naive_bayes_is_suspicious(mock_nb, mock_gemini):
    """
    1차 나이브 베이즈가 SAFE 기준을 넘는 위험도로 판정하면 Gemini 2차 검증을 호출하고,
    1차 나이브 베이즈 점수도 stage1_naive_bayes로 함께 실어보내야 한다.
    """
    mock_nb.return_value = _nb_result("DANGEROUS", 91)
    mock_gemini.return_value = {
        "is_mock": False,
        "result": {"grade": "DANGEROUS", "risk_score": 90, "tone_analysis": "", "evidence": [], "reason": ""}
    }

    service = SmishingAnalysisService()
    text_analysis, naive_bayes_score, llm_available = await service._analyze_text_hybrid(
        "[국민건강보험] 즉시 확인하세요 http://bit.ly/fake"
    )

    mock_gemini.assert_awaited_once()
    assert text_analysis["result"]["risk_score"] == 90
    assert text_analysis["stage1_naive_bayes"]["risk_score"] == 91
    assert naive_bayes_score == 91
    assert llm_available is True


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_gemini", new_callable=AsyncMock)
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_hybrid_text_track_falls_back_to_gemini_when_naive_bayes_unavailable(mock_nb, mock_gemini):
    """
    나이브 베이즈 모델 로드에 실패한 경우, SAFE 판정 여부와 무관하게
    안전하게 Gemini 2차 검증으로 폴백해야 한다 (fail-safe). 이 경우 나이브 베이즈 점수는
    스코어링에 반영할 수 없으므로 None으로 전달되어야 한다.
    """
    mock_nb.return_value = _nb_result("UNKNOWN", 0, is_available=False)
    mock_gemini.return_value = {
        "is_mock": False,
        "result": {"grade": "SAFE", "risk_score": 5, "tone_analysis": "", "evidence": [], "reason": ""}
    }

    service = SmishingAnalysisService()
    text_analysis, naive_bayes_score, llm_available = await service._analyze_text_hybrid("테스트 메시지")

    mock_gemini.assert_awaited_once()
    assert text_analysis["result"]["risk_score"] == 5
    assert naive_bayes_score is None
    assert llm_available is True


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_gemini", new_callable=AsyncMock)
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_hybrid_text_track_marks_llm_unavailable_when_gemini_errors(mock_nb, mock_gemini):
    """
    나이브 베이즈가 의심 판정해 에스컬레이션했지만 Gemini 호출 자체가 실패(UNKNOWN)한 경우,
    llm_available=False로 표시되어 스코어링 단계에서 나이브 베이즈 점수를 fail-safe로 신뢰하도록 해야 한다.
    """
    mock_nb.return_value = _nb_result("DANGEROUS", 91)
    mock_gemini.return_value = {
        "is_mock": False,
        "result": {"grade": "UNKNOWN", "risk_score": 0, "tone_analysis": "", "evidence": [], "error_message": "Rate Limit"}
    }

    service = SmishingAnalysisService()
    text_analysis, naive_bayes_score, llm_available = await service._analyze_text_hybrid(
        "[국민건강보험] 즉시 확인하세요 http://bit.ly/fake"
    )

    assert naive_bayes_score == 91
    assert llm_available is False


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
@patch("app.analysis.service.trace_url", new_callable=AsyncMock)
async def test_analyze_pipeline_forces_high_when_local_domain_rule_matches(mock_trace, mock_nb):
    """
    로컬 도메인 룰(.ru 등)이 매치되면 텍스트 문맥 점수가 아무리 낮아도(나이브 베이즈 SAFE라
    Gemini조차 스킵된 상황) 확정 악성으로 승격되어 최종 등급이 HIGH로 강제되어야 한다.
    """
    mock_nb.return_value = _nb_result("SAFE", 5)
    mock_trace.return_value = "https://malicious.ru/phish"

    service = SmishingAnalysisService()
    service.url_analyzer.scan_url = AsyncMock(return_value={
        "is_malicious": False,
        "url_risk_score": 0.0,
        "source": "Hybrid-Engine (GSB+VT)",
        "error_message": None,
        "is_gsb_confirmed": False,
        "is_vt_confirmed": False
    })

    result = await service.analyze_pipeline("평범한 문자입니다 https://bit.ly/xyz")

    assert result.risk_grade == "HIGH"
    assert result.final_score >= 70
    assert result.rule_analysis["has_malicious_domain_pattern"] is True


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_gemini", new_callable=AsyncMock)
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_analyze_pipeline_uses_available_zero_score_rules_when_text_engines_are_down(mock_nb, mock_gemini):
    """
    나이브 베이즈 모델 로드 실패 + Gemini 호출도 동시에 실패(rate limit 등)하는 경우,
    URL/규칙 신호가 전혀 없는 문자라도 최종 등급이 조용히 LOW로 나와선 안 된다.
    두 분류기가 동시에 다운됐다는 인프라 장애가 "안전 확인됨"으로 둔갑하면 안 됨 (fail-open 방지).
    """
    mock_nb.return_value = _nb_result("UNKNOWN", 0, is_available=False)
    mock_gemini.return_value = {
        "is_mock": False,
        "result": {"grade": "UNKNOWN", "risk_score": 0, "tone_analysis": "", "evidence": [], "error_message": "Rate Limit"}
    }

    service = SmishingAnalysisService()
    result = await service.analyze_pipeline("URL도 없고 특이사항도 없는 문자")

    assert result.status == "SUCCESS"
    assert result.rule_analysis["rule_score"] == 0
    assert result.risk_grade == "LOW"
    assert result.final_score == 0


@pytest.mark.asyncio
@patch("app.analysis.service.analyze_text_with_naive_bayes", new_callable=AsyncMock)
async def test_analyze_pipeline_does_not_fail_open_when_pipeline_itself_throws(mock_nb):
    """
    나이브 베이즈/Gemini 개별 실패가 아니라 파이프라인 자체가 처리 중 예외로 죽는 경우
    (네트워크 오류, 버그 등)에도 status="ERROR"만 보고 걸러내지 않는 소비자를 위해
    risk_grade/final_score 자체가 조용히 LOW/0("안전 확인됨")으로 나와선 안 된다.
    """
    mock_nb.side_effect = RuntimeError("예기치 못한 파이프라인 장애")

    service = SmishingAnalysisService()
    result = await service.analyze_pipeline("URL도 없고 특이사항도 없는 문자")

    assert result.status == "ERROR"
    assert result.risk_grade != "LOW"
    assert result.final_score >= 40
