"""통합 분석 서비스의 하이브리드 텍스트 연결 테스트"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.analysis.service import SmishingAnalysisService


def _text_analysis(
    *,
    self_model_score: int | None,
    selected_score: int | None,
    gemini_called: bool,
    gemini_available: bool,
    decision_source: str,
) -> dict:
    """서비스 테스트에서 사용할 하이브리드 텍스트 결과 생성"""

    grade = "UNKNOWN" if selected_score is None else (
        "DANGEROUS" if selected_score >= 70 else "SAFE"
    )

    return {
        "engine": "hybrid_stacking_gemini",
        "result": {
            "grade": grade,
            "risk_score": selected_score,
            "tone_analysis": "",
            "evidence": [],
            "reason": "테스트 분석 결과",
            "error_message": (
                "ALL_TEXT_ENGINES_UNAVAILABLE"
                if selected_score is None
                else None
            ),
        },
        "self_model": {
            "risk_score": self_model_score,
            "risk_probability": (
                None
                if self_model_score is None
                else self_model_score / 100
            ),
            "confidence": 0.8 if self_model_score is not None else 0.0,
        },
        "gemini": None,
        "gemini_called": gemini_called,
        "gemini_available": gemini_available,
        "decision_source": decision_source,
        "routing_decision": "TEST_DECISION",
        "routing_reason": "TEST_REASON",
        "fallback_applied": decision_source == "STACKING_FALLBACK",
    }


def _rule_result(
    _text: str,
    _traced_url: str | None,
) -> dict:
    """추가 위험 신호가 없는 정상 규칙 분석 결과"""

    return {
        "rule_score": 0,
        "has_malicious_domain_pattern": False,
        "matched_rules": [],
        "error_message": None,
    }


@pytest.mark.asyncio
async def test_pipeline_uses_stacking_result_when_gemini_is_skipped() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=10,
                selected_score=10,
                gemini_called=False,
                gemini_available=False,
                decision_source="STACKING",
            )
        )
    )
    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=_rule_result,
    )

    result = await service.analyze_pipeline("오늘 저녁 같이 먹자")

    text_analyzer.analyze.assert_awaited_once_with(
        "오늘 저녁 같이 먹자",
        force_gemini=False,
    )
    assert result.status == "SUCCESS"
    assert result.text_analysis["decision_source"] == "STACKING"
    assert result.text_analysis["gemini_called"] is False
    assert result.url_analysis is None
    assert result.risk_grade == "LOW"


@pytest.mark.asyncio
async def test_pipeline_combines_stacking_and_successful_gemini() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=50,
                selected_score=90,
                gemini_called=True,
                gemini_available=True,
                decision_source="GEMINI",
            )
        )
    )
    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=_rule_result,
    )

    result = await service.analyze_pipeline("본인 확인이 필요합니다")

    assert result.status == "SUCCESS"
    assert result.text_analysis["decision_source"] == "GEMINI"
    assert result.text_analysis["gemini_available"] is True
    assert result.final_score >= 40


@pytest.mark.asyncio
async def test_pipeline_forces_gemini_when_rule_score_is_high() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=10,
                selected_score=80,
                gemini_called=True,
                gemini_available=True,
                decision_source="GEMINI",
            )
        )
    )

    def suspicious_rule_result(
        _text: str,
        _traced_url: str | None,
    ) -> dict:
        return {
            "rule_score": 40,
            "has_malicious_domain_pattern": False,
            "matched_rules": ["institution_pattern"],
            "error_message": None,
        }

    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=suspicious_rule_result,
    )

    result = await service.analyze_pipeline(
        "기관 사칭 의심 문자"
    )

    text_analyzer.analyze.assert_awaited_once_with(
        "기관 사칭 의심 문자",
        force_gemini=True,
    )
    assert result.status == "SUCCESS"


@pytest.mark.asyncio
async def test_pipeline_forces_gemini_when_rule_preview_fails() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=10,
                selected_score=75,
                gemini_called=True,
                gemini_available=True,
                decision_source="GEMINI",
            )
        )
    )

    def raising_rule_analyzer(
        _text: str,
        _traced_url: str | None,
    ) -> dict:
        raise RuntimeError("sensitive rule detail")

    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=raising_rule_analyzer,
    )

    result = await service.analyze_pipeline(
        "규칙 분석 실패 문자"
    )

    text_analyzer.analyze.assert_awaited_once_with(
        "규칙 분석 실패 문자",
        force_gemini=True,
    )
    assert result.status == "SUCCESS"
    assert result.rule_analysis["error_message"] == (
        "RULE_ANALYSIS_FAILED"
    )


@pytest.mark.asyncio
@patch("app.analysis.service.trace_url", new_callable=AsyncMock)
async def test_pipeline_forces_high_when_local_domain_rule_matches(
    mock_trace: AsyncMock,
) -> None:
    mock_trace.return_value = "https://malicious.ru/phish"
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=5,
                selected_score=5,
                gemini_called=False,
                gemini_available=False,
                decision_source="STACKING",
            )
        )
    )
    service = SmishingAnalysisService(text_analyzer=text_analyzer)
    service.url_analyzer.scan_url = AsyncMock(
        return_value={
            "is_malicious": False,
            "url_risk_score": 0.0,
            "source": "Hybrid-Engine (GSB+VT)",
            "available": True,
            "failed_providers": [],
            "pending_providers": [],
            "provider_error_codes": {},
            "error_message": None,
            "is_gsb_confirmed": False,
            "is_vt_confirmed": False,
        }
    )

    result = await service.analyze_pipeline(
        "평범한 문자입니다 https://bit.ly/xyz"
    )

    assert result.risk_grade == "HIGH"
    assert result.final_score >= 70
    assert result.rule_analysis["has_malicious_domain_pattern"] is True


@pytest.mark.asyncio
async def test_pipeline_does_not_fail_open_when_text_engines_are_down() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            return_value=_text_analysis(
                self_model_score=None,
                selected_score=None,
                gemini_called=True,
                gemini_available=False,
                decision_source="UNAVAILABLE",
            )
        )
    )
    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=_rule_result,
    )

    result = await service.analyze_pipeline("URL이 없는 테스트 문자")

    assert result.status == "SUCCESS"
    assert result.rule_analysis["rule_score"] == 0
    assert result.risk_grade in {"MEDIUM", "HIGH"}
    assert result.final_score >= 40


@pytest.mark.asyncio
async def test_pipeline_does_not_fail_open_when_text_analyzer_raises() -> None:
    text_analyzer = SimpleNamespace(
        analyze=AsyncMock(
            side_effect=RuntimeError("expected pipeline failure")
        )
    )
    service = SmishingAnalysisService(
        text_analyzer=text_analyzer,
        rule_analyzer=_rule_result,
    )

    result = await service.analyze_pipeline("테스트 문자")

    assert result.status == "ERROR"
    assert result.risk_grade != "LOW"
    assert result.final_score >= 40
