from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisResultEvent,
    AnalysisResultPayload,
    RawScores,
    WeightedContributions,
)


def test_completed_event_is_valid() -> None:
    """정상 완료(COMPLETED) 이벤트 직렬화 및 직렬화 복원 테스트"""
    event = AnalysisResultEvent(
        schemaVersion="1.0",
        eventId=uuid4(),
        causationId=uuid4(),
        analysisId=1,
        clientMessageId="message-1",
        traceId=uuid4(),
        occurredAt=datetime.now(timezone.utc),
        eventType=AnalysisEventType.COMPLETED,
        payload=AnalysisResultPayload(
            finalScore=74,
            riskGrade="HIGH",
            phishingType=None,
            rawScores=RawScores(
                text=80,
                url=50,
                rules=20,
            ),
            weightedContributions=WeightedContributions(
                text=52,
                url=15,
                rules=7,
            ),
        ),
    )

    restored = AnalysisResultEvent.model_validate_json(
        event.model_dump_json()
    )

    assert restored == event
    assert restored.payload.phishingType is None


def test_partial_event_requires_failed_tracks() -> None:
    """부분 성공(PARTIAL) 이벤트 시 failedTracks 누락을 검증 테스트"""
    with pytest.raises(ValidationError):
        AnalysisResultEvent(
            schemaVersion="1.0",
            eventId=uuid4(),
            causationId=uuid4(),
            analysisId=1,
            traceId=uuid4(),
            occurredAt=datetime.now(timezone.utc),
            eventType=AnalysisEventType.PARTIAL,
            payload=AnalysisResultPayload(
                finalScore=60,
                riskGrade="MEDIUM",
                rawScores=RawScores(text=70),
                failedTracks=[],
            ),
        )


def test_failed_event_requires_failure_code() -> None:
    """분석 실패(FAILED) 이벤트 시 failureCode 누락 검증 테스트"""
    with pytest.raises(ValidationError):
        AnalysisResultEvent(
            schemaVersion="1.0",
            eventId=uuid4(),
            causationId=uuid4(),
            analysisId=1,
            traceId=uuid4(),
            occurredAt=datetime.now(timezone.utc),
            eventType=AnalysisEventType.FAILED,
            payload=AnalysisResultPayload(
                rawScores=RawScores(),
            ),
        )


def test_failed_event_cannot_be_low_risk() -> None:
    """분석 실패(FAILED) 이벤트에 LOW 위험 등급 설정 금지 검증 테스트"""
    with pytest.raises(ValidationError):
        AnalysisResultEvent(
            schemaVersion="1.0",
            eventId=uuid4(),
            causationId=uuid4(),
            analysisId=1,
            traceId=uuid4(),
            occurredAt=datetime.now(timezone.utc),
            eventType=AnalysisEventType.FAILED,
            payload=AnalysisResultPayload(
                riskGrade="LOW",
                rawScores=RawScores(),
                failureCode="ALL_TRACKS_FAILED",
            ),
        )