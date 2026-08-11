from app.analysis.execution import (
    AnalysisExecutionStatus,
    classify_execution,
)
from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
)


def _result(
    *,
    status: str = "SUCCESS",
    text_analysis: dict | None = None,
    url_analysis: dict | None = None,
    rule_analysis: dict | None = None,
) -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status=status,
        message="analysis result",
        final_score=50,
        risk_grade=RiskGrade.MEDIUM,
        contribution_breakdown=ContributionBreakdown(
            llm=30,
            hybrid_url=10,
            rules=10,
        ),
        text_analysis=text_analysis,
        url_analysis=url_analysis,
        rule_analysis=rule_analysis,
    )


def _text_analysis(
    *,
    score: int | None = 30,
    self_model_score: int | None = 30,
    gemini_called: bool = False,
    gemini_available: bool = False,
    error_message: str | None = None,
) -> dict:
    """현재 하이브리드 텍스트 응답 스키마로 테스트 데이터를 만든다."""

    return {
        "result": {
            "grade": "UNKNOWN" if score is None else "SAFE",
            "risk_score": score,
            "error_message": error_message,
        },
        "self_model": {
            "risk_score": self_model_score,
        },
        "gemini_called": gemini_called,
        "gemini_available": gemini_available,
    }


def test_classifies_successful_execution_as_completed() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(),
            url_analysis={
                "available": True,
                "failed_providers": [],
            },
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.COMPLETED
    assert execution.failed_tracks == ()


def test_classifies_provider_failure_as_partial() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(),
            url_analysis={
                "available": True,
                "failed_providers": ["GSB"],
            },
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("URL:GSB",)


def test_classifies_unavailable_url_track_as_partial() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(),
            url_analysis={
                "available": False,
                "failed_providers": [
                    "GSB",
                    "VIRUSTOTAL",
                ],
            },
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("URL",)


def test_classifies_pipeline_error_as_failed() -> None:
    execution = classify_execution(_result(status="ERROR"))

    assert execution.status == AnalysisExecutionStatus.FAILED
    assert execution.failed_tracks == ("PIPELINE",)


def test_classifies_gemini_failure_with_valid_stacking() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(
                score=70,
                self_model_score=70,
                gemini_called=True,
                gemini_available=False,
            ),
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("TEXT:GEMINI",)


def test_classifies_stacking_failure_with_valid_gemini() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(
                score=20,
                self_model_score=None,
                gemini_called=True,
                gemini_available=True,
            ),
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("TEXT:STACKING",)


def test_classifies_all_text_engines_unavailable() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(
                score=None,
                self_model_score=None,
                gemini_called=True,
                gemini_available=False,
                error_message="ALL_TEXT_ENGINES_UNAVAILABLE",
            ),
            rule_analysis={"error_message": None},
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("TEXT",)


def test_classifies_rule_failure_as_partial() -> None:
    execution = classify_execution(
        _result(
            text_analysis=_text_analysis(),
            rule_analysis={
                "error_message": "RULE_ANALYSIS_FAILED",
            },
        )
    )

    assert execution.status == AnalysisExecutionStatus.PARTIAL
    assert execution.failed_tracks == ("RULES",)
