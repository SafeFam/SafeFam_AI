from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.analysis.execution import (
    AnalysisExecution,
    AnalysisExecutionStatus,
)
from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
)
from app.infrastructure.rabbitmq.result_factory import (
    AnalysisResultEventFactory,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisRequestedEvent,
)


def _request() -> AnalysisRequestedEvent:
    return AnalysisRequestedEvent.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": str(uuid4()),
            "analysisId": 123,
            "clientMessageId": "sms-result-001",
            "traceId": str(uuid4()),
            "occurredAt": datetime.now(timezone.utc),
            "payload": {
                "sender": "1588-0000",
                "content": "분석 대상 문자",
                "receivedAt": datetime.now(timezone.utc),
                "source": "AUTO",
            },
        }
    )


def _result(
    *,
    status: str = "SUCCESS",
) -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status=status,
        message="analysis result",
        final_score=82 if status == "SUCCESS" else 40,
        risk_grade=(
            RiskGrade.HIGH
            if status == "SUCCESS"
            else RiskGrade.MEDIUM
        ),
        contribution_breakdown=ContributionBreakdown(
            llm=42,
            hybrid_url=25,
            rules=15,
        ),
        text_analysis=(
            {
                "result": {
                    "risk_score": 84,
                    "grade": "DANGEROUS",
                    "reason": "금융기관 사칭",
                    "evidence": ["인증 요구"],
                },
                "stage1_naive_bayes": {
                    "risk_score": 70,
                    "grade": "SUSPICIOUS",
                },
            }
            if status == "SUCCESS"
            else None
        ),
        url_analysis=(
            {
                "has_url": True,
                "original_url": "https://short.example/a",
                "origin_url": "https://example.test/login",
                "is_url_malicious": True,
                "url_risk_score": 0.9,
                "engine_source": "Hybrid-Engine",
                "error_message": None,
            }
            if status == "SUCCESS"
            else None
        ),
        rule_analysis=(
            {
                "rule_score": 75,
                "matched_rules": ["financial impersonation"],
                "has_malicious_domain_pattern": True,
            }
            if status == "SUCCESS"
            else None
        ),
    )


@pytest.mark.parametrize(
    ("status", "event_type", "failed_tracks"),
    [
        (
            AnalysisExecutionStatus.COMPLETED,
            AnalysisEventType.COMPLETED,
            (),
        ),
        (
            AnalysisExecutionStatus.PARTIAL,
            AnalysisEventType.PARTIAL,
            ("URL:VIRUSTOTAL",),
        ),
    ],
)
def test_factory_maps_successful_execution(
    status: AnalysisExecutionStatus,
    event_type: AnalysisEventType,
    failed_tracks: tuple[str, ...],
) -> None:
    request = _request()
    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=status,
            result=_result(),
            failed_tracks=failed_tracks,
        ),
    )

    assert event.eventType == event_type
    assert event.causationId == request.eventId
    assert event.analysisId == request.analysisId
    assert event.traceId == request.traceId
    assert event.payload.finalScore == 82
    assert event.payload.riskGrade == "HIGH"
    assert event.payload.rawScores.text == 84
    assert event.payload.rawScores.url == 90
    assert event.payload.rawScores.rules == 75
    assert event.payload.failedTracks == list(failed_tracks)


def test_factory_maps_failed_execution_without_message_content() -> None:
    request = _request()
    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.FAILED,
            result=_result(status="ERROR"),
            failed_tracks=("PIPELINE",),
        ),
    )

    assert event.eventType == AnalysisEventType.FAILED
    assert event.payload.finalScore is None
    assert event.payload.riskGrade is None
    assert event.payload.failureCode == "PIPELINE_FAILED"
    assert request.payload.content not in event.model_dump_json()
