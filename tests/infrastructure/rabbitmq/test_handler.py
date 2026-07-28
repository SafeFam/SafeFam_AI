import logging
from unittest.mock import AsyncMock

import pytest

from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
)
from app.infrastructure.rabbitmq.handler import (
    AnalysisRequestHandler,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
)


def create_analysis_requested_event() -> AnalysisRequestedEvent:
    return AnalysisRequestedEvent.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": (
                "1fb898fa-d89d-4d0b-a43f-a8b00daeb765"
            ),
            "analysisId": 123,
            "clientMessageId": "sms-20260728-001",
            "traceId": (
                "2d59c74e-0691-4f01-bde3-c657ba4c90cd"
            ),
            "occurredAt": "2026-07-28T01:30:00Z",
            "payload": {
                "sender": "1588-0000",
                "content": (
                    "[국민은행] 계좌가 정지되었습니다."
                ),
                "receivedAt": (
                    "2026-07-28T10:29:00+09:00"
                ),
                "source": "AUTO",
            },
        }
    )


def create_success_result() -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status="SUCCESS",
        message="Analysis completed successfully.",
        final_score=82,
        risk_grade=RiskGrade.HIGH,
        contribution_breakdown=ContributionBreakdown(
            llm=42,
            hybrid_url=25,
            rules=15,
        ),
        text_analysis={
            "result": {
                "risk_score": 84,
                "grade": "DANGEROUS",
            }
        },
        url_analysis={
            "has_url": True,
            "is_url_malicious": True,
        },
        rule_analysis={
            "rule_score": 75,
            "matched_rules": ["financial impersonation"],
        },
    )


def create_error_result() -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status="ERROR",
        message="Gemini API timeout",
        final_score=40,
        risk_grade=RiskGrade.MEDIUM,
        contribution_breakdown=ContributionBreakdown(
            llm=0,
            hybrid_url=0,
            rules=0,
        ),
        text_analysis=None,
        url_analysis=None,
        rule_analysis=None,
    )

@pytest.mark.asyncio
async def test_handler_passes_event_content_to_analysis_pipeline():
    """정상 분석 테스트"""
    event = create_analysis_requested_event()
    expected_result = create_success_result()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.return_value = (
        expected_result
    )

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    actual_result = await handler.handle(event)

    analysis_service.analyze_pipeline.assert_awaited_once_with(
        event.payload.content
    )
    assert actual_result is expected_result

@pytest.mark.asyncio
async def test_handler_returns_pipeline_error_result():
    """실패 변환 테스트"""
    event = create_analysis_requested_event()
    error_result = create_error_result()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.return_value = (
        error_result
    )

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    actual_result = await handler.handle(event)

    assert actual_result is error_result

    analysis_service.analyze_pipeline.assert_awaited_once_with(
        event.payload.content
    )

@pytest.mark.asyncio
async def test_handler_propagates_analysis_service_exception():
    """분석 서비스 예외 전달 테스트"""
    event = create_analysis_requested_event()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.side_effect = RuntimeError(
        "Unexpected pipeline failure"
    )

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected pipeline failure",
    ):
        await handler.handle(event)

@pytest.mark.asyncio
async def test_handler_logs_event_tracking_identifiers(
    caplog: pytest.LogCaptureFixture,
):
    """추적 식별자 로그 테스트"""
    event = create_analysis_requested_event()
    expected_result = create_success_result()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.return_value = (
        expected_result
    )

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    with caplog.at_level(logging.INFO):
        await handler.handle(event)

    log_output = caplog.text

    assert str(event.eventId) in log_output
    assert str(event.analysisId) in log_output
    assert str(event.traceId) in log_output
    assert str(expected_result.final_score) in log_output
    assert str(expected_result.risk_grade) in log_output

@pytest.mark.asyncio
async def test_handler_does_not_log_message_content(
    caplog: pytest.LogCaptureFixture,
):
    """문자 원문을 로그에 남기지 않는지 테스트"""
    event = create_analysis_requested_event()
    expected_result = create_success_result()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.return_value = (
        expected_result
    )

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    with caplog.at_level(logging.INFO):
        await handler.handle(event)

    assert event.payload.content not in caplog.text

@pytest.mark.asyncio
async def test_handler_logs_tracking_identifiers_on_failure(
    caplog: pytest.LogCaptureFixture,
):
    """실패 로그 식별자 테스트"""
    event = create_analysis_requested_event()
    error_result = create_error_result()

    analysis_service = AsyncMock()
    analysis_service.analyze_pipeline.return_value = error_result

    handler = AnalysisRequestHandler(
        analysis_service=analysis_service
    )

    with caplog.at_level(logging.WARNING):
        actual_result = await handler.handle(event)

    assert actual_result is error_result
    assert str(event.eventId) in caplog.text
    assert str(event.analysisId) in caplog.text
    assert str(event.traceId) in caplog.text
