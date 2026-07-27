import pytest
from unittest.mock import patch
from app.service.security.gemini_text_analyzer import determine_text_risk_grade, analyze_text_with_gemini


def test_determine_text_risk_grade_safe():
    assert determine_text_risk_grade(10) == "SAFE"


def test_determine_text_risk_grade_suspicious():
    assert determine_text_risk_grade(55) == "SUSPICIOUS"


def test_determine_text_risk_grade_dangerous():
    assert determine_text_risk_grade(90) == "DANGEROUS"


@pytest.mark.asyncio
@patch("app.service.security.gemini_text_analyzer.MOCK_ENABLED", True)
async def test_analyze_text_with_gemini_mock_mode():
    """
    Mock 스위치가 켜졌을 때 실제 Gemini API를 호출하는 대신 모듈에 내장된 사전 정의 데이터로
    우회 처리되어 어조/근거 분석 결과가 정상적으로 가공되는지 검증합니다.
    """
    text = "[검찰청] 귀하 명의로 대포통장이 개설되어 수사가 진행 중입니다. 즉시 아래 링크로 접속하여 신원을 확인하세요."

    result = await analyze_text_with_gemini(text)

    assert result["is_mock"] is True
    assert result["result"]["grade"] == "DANGEROUS"
    assert result["result"]["risk_score"] == 85
    assert len(result["result"]["evidence"]) > 0


@pytest.mark.asyncio
@patch("app.service.security.gemini_text_analyzer.MOCK_ENABLED", True)
async def test_analyze_text_with_gemini_mock_default_safe():
    """
    Mock 모드는 텍스트 내용과 무관하게 항상 동일한 내장 데모 데이터를 반환하므로,
    Mock 스위치가 켜져 있으면 어떤 입력이든 그 고정 결과로 처리되는지 검증합니다.
    """
    text = "정의되지 않은 임의의 문자 메시지입니다."

    result = await analyze_text_with_gemini(text)

    assert result["is_mock"] is True
    assert result["result"]["grade"] == "DANGEROUS"


@pytest.mark.asyncio
@patch("app.service.security.gemini_text_analyzer.MOCK_ENABLED", False)
@patch("app.service.security.gemini_text_analyzer.GEMINI_API_KEY", None)
async def test_analyze_text_with_gemini_missing_api_key():
    """
    API 키가 누락된 상태에서 실제 API를 호출하지 않고, fail-closed 정책에 따라
    SAFE가 아닌 UNKNOWN(판정 불가) 등급 + 에러 메시지를 반환하는지 검증합니다.
    """
    text = "테스트 메시지"

    result = await analyze_text_with_gemini(text)

    assert result["is_mock"] is False
    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["error_message"] == "Missing API Key"
