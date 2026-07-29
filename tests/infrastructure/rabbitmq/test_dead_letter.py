import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from aio_pika import DeliveryMode

from app.infrastructure.errors import (
    RetryableProcessingError,
)
from app.infrastructure.rabbitmq.dead_letter import (
    DeadLetterPublisher,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
)


SENSITIVE_CONTENT = (
    "[국민은행] 계좌가 정지되었습니다. "
    "https://malicious.example/login"
)


def create_settings(
    *,
    timeout: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY=(
            "analysis.requested.dead.v1"
        ),
        RABBITMQ_PUBLISH_TIMEOUT_SECONDS=timeout,
    )


def create_request_event() -> AnalysisRequestedEvent:
    return AnalysisRequestedEvent.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": str(uuid4()),
            "analysisId": 123,
            "clientMessageId": "sms-dlq-001",
            "traceId": str(uuid4()),
            "occurredAt": datetime.now(timezone.utc),
            "payload": {
                "sender": "1588-0000",
                "content": SENSITIVE_CONTENT,
                "receivedAt": datetime.now(timezone.utc),
                "source": "AUTO",
            },
        }
    )


@pytest.mark.asyncio
async def test_publish_sends_sanitized_persistent_event():
    """DLQ 메시지에는 추적 정보만 포함하고 원문은 제외합니다."""
    exchange = AsyncMock()
    publisher = DeadLetterPublisher(
        exchange=exchange,
        app_settings=create_settings(),
    )
    request_event = create_request_event()

    dead_letter_event = await publisher.publish(
        original_message_id="rabbit-message-001",
        failure_code="PROCESSING_RETRIES_EXHAUSTED",
        request_event=request_event,
    )

    exchange.publish.assert_awaited_once()
    message = exchange.publish.await_args.args[0]
    arguments = exchange.publish.await_args.kwargs

    assert (
        arguments["routing_key"]
        == "analysis.requested.dead.v1"
    )
    assert arguments["mandatory"] is True
    assert message.delivery_mode == DeliveryMode.PERSISTENT
    assert message.content_type == "application/json"
    assert message.message_id == str(
        dead_letter_event.eventId
    )
    assert message.correlation_id == str(
        request_event.traceId
    )
    assert message.headers["sanitized"] is True

    body_text = message.body.decode("utf-8")
    body = json.loads(body_text)

    assert body["analysisId"] == 123
    assert body["traceId"] == str(
        request_event.traceId
    )
    assert (
        body["failureCode"]
        == "PROCESSING_RETRIES_EXHAUSTED"
    )
    assert "payload" not in body
    assert "content" not in body
    assert "sender" not in body
    assert "originalUrl" not in body
    assert SENSITIVE_CONTENT not in body_text
    assert request_event.payload.sender not in body_text


@pytest.mark.asyncio
async def test_publish_without_valid_request_uses_no_analysis_data():
    """파싱 불가능한 원본은 메시지 ID와 실패 코드만 남깁니다."""
    exchange = AsyncMock()
    publisher = DeadLetterPublisher(
        exchange=exchange,
        app_settings=create_settings(),
    )

    event = await publisher.publish(
        original_message_id="rabbit-invalid-001",
        failure_code="INVALID_JSON",
        request_event=None,
    )

    assert event.originalMessageId == "rabbit-invalid-001"
    assert event.analysisId is None
    assert event.traceId is None


@pytest.mark.asyncio
async def test_publish_timeout_becomes_retryable_error():
    """DLQ timeout은 원본 재전달을 위한 재시도 오류로 변환합니다."""
    async def delayed_publish(*args, **kwargs):
        await asyncio.sleep(0.1)

    exchange = AsyncMock()
    exchange.publish.side_effect = delayed_publish

    publisher = DeadLetterPublisher(
        exchange=exchange,
        app_settings=create_settings(timeout=0.01),
    )

    with pytest.raises(
        RetryableProcessingError,
        match="publication timed out",
    ) as error:
        await publisher.publish(
            original_message_id="rabbit-message-001",
            failure_code="INVALID_JSON",
            request_event=None,
        )

    assert error.value.failure_code == "DLQ_PUBLISH_TIMEOUT"


@pytest.mark.asyncio
async def test_publish_failure_becomes_retryable_error():
    """브로커 발행 실패는 재시도 가능한 오류로 변환합니다."""
    exchange = AsyncMock()
    exchange.publish.side_effect = RuntimeError(
        "RabbitMQ unavailable"
    )

    publisher = DeadLetterPublisher(
        exchange=exchange,
        app_settings=create_settings(),
    )

    with pytest.raises(
        RetryableProcessingError,
        match="Failed to publish",
    ) as error:
        await publisher.publish(
            original_message_id="rabbit-message-001",
            failure_code="INVALID_JSON",
            request_event=None,
        )

    assert error.value.failure_code == "DLQ_PUBLISH_FAILED"
