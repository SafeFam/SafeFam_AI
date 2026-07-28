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


def test_classifies_successful_execution_as_completed() -> None:
    execution = classify_execution(
        _result(
            text_analysis={
                "result": {
                    "grade": "SAFE",
                    "error_message": None,
                },
            },
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
            text_analysis={
                "result": {
                    "grade": "SAFE",
                    "error_message": None,
                },
            },
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
            text_analysis={
                "result": {
                    "grade": "SAFE",
                    "error_message": None,
                },
            },
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
    execution = classify_execution(
        _result(status="ERROR")
    )

    assert execution.status == AnalysisExecutionStatus.FAILED
    assert execution.failed_tracks == ("PIPELINE",)
