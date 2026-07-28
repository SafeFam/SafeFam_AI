import json
from unittest.mock import AsyncMock, Mock

import pytest

from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
)
from app.infrastructure.errors import (
    NonRetryableProcessingError,
)
from app.infrastructure.rabbitmq.consumer import (
    AnalysisRequestConsumer,
)

def create_valid_message_body() -> bytes:
    event_data = {
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

    return json.dumps(
        event_data,
        ensure_ascii=False,
    ).encode("utf-8")

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
        text_analysis=None,
        url_analysis=None,
        rule_analysis=None,
    )


def create_error_result() -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status="ERROR",
        message="All analysis tracks failed.",
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


def create_message(
    *,
    body: bytes | None = None,
    redelivered: bool = False,
):
    message = AsyncMock()

    message.body = body or create_valid_message_body()
    message.message_id = "rabbit-message-001"
    message.redelivered = redelivered

    return message

def create_consumer():
    request_queue = AsyncMock()
    handler = AsyncMock()

    result_publisher = Mock()
    result_publisher.publish = AsyncMock()

    result_factory = Mock()
    result_factory.create.return_value = Mock(
        eventId="result-event-id",
        eventType=Mock(
            value="ANALYSIS_COMPLETED"
        ),
    )

    dead_letter_publisher = Mock()
    dead_letter_publisher.publish = AsyncMock()

    consumer = AnalysisRequestConsumer(
        request_queue=request_queue,
        handler=handler,
        result_publisher=result_publisher,
        result_factory=result_factory,
        dead_letter_publisher=dead_letter_publisher,
    )

    return consumer, request_queue, handler

@pytest.mark.asyncio
async def test_consumer_acknowledges_successful_message():
    """정상 처리 ACK 테스트"""
    consumer, _, handler = create_consumer()
    message = create_message()

    expected_result = create_success_result()
    handler.handle.return_value = expected_result

    await consumer._on_message(message)

    handler.handle.assert_awaited_once()

    handled_event = handler.handle.await_args.args[0]
    assert handled_event.analysisId == 123
    assert handled_event.payload.content == (
        "[국민은행] 계좌가 정지되었습니다."
    )

    consumer.result_factory.create.assert_called_once()
    factory_arguments = (
        consumer.result_factory.create.call_args.kwargs
    )
    assert factory_arguments["request"] is handled_event
    assert (
        factory_arguments["execution"].status.value
        == "COMPLETED"
    )

    result_event = (
        consumer.result_factory.create.return_value
    )
    consumer.result_publisher.publish.assert_awaited_once_with(
        result_event
    )

    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()
    message.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumer_does_not_ack_when_publication_fails():
    """결과 이벤트 발행 실패 시 요청 메시지를 ACK X"""
    consumer, _, handler = create_consumer()
    message = create_message()

    handler.handle.return_value = create_success_result()
    consumer.result_publisher.publish.side_effect = (
        RuntimeError("RabbitMQ publish failed")
    )

    await consumer._on_message(message)

    consumer.result_publisher.publish.assert_awaited_once()
    message.ack.assert_not_awaited()
    message.nack.assert_awaited_once_with(requeue=True)
    message.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumer_publishes_failed_result_and_acks():
    """ERROR 응답을 FAILED 결과 이벤트로 발행한 뒤 ACK"""
    consumer, _, handler = create_consumer()
    message = create_message()

    handler.handle.return_value = create_error_result()

    await consumer._on_message(message)

    factory_arguments = (
        consumer.result_factory.create.call_args.kwargs
    )
    assert (
        factory_arguments["execution"].status.value
        == "FAILED"
    )
    consumer.result_publisher.publish.assert_awaited_once()
    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_routes_invalid_json_to_sanitized_dlq():
    """잘못된 JSON은 정제된 DLQ 이벤트 발행 후 ACK합니다."""
    consumer, _, handler = create_consumer()
    message = create_message(body=b"{invalid-json")

    await consumer._on_message(message)

    handler.handle.assert_not_awaited()
    consumer.dead_letter_publisher.publish.assert_awaited_once_with(
        original_message_id=message.message_id,
        failure_code="INVALID_JSON",
        request_event=None,
    )
    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()
    message.reject.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_routes_unsupported_schema_to_dlq():
    """지원하지 않는 버전은 정제된 DLQ 이벤트로 격리합니다."""
    consumer, _, handler = create_consumer()

    invalid_event = {
        "schemaVersion": "2.0",
        "eventId": "not-a-uuid",
        "analysisId": 0,
    }

    message = create_message(
        body=json.dumps(invalid_event).encode("utf-8")
    )

    await consumer._on_message(message)

    handler.handle.assert_not_awaited()
    consumer.dead_letter_publisher.publish.assert_awaited_once_with(
        original_message_id=message.message_id,
        failure_code="UNSUPPORTED_SCHEMA_VERSION",
        request_event=None,
    )
    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()
    message.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumer_routes_invalid_event_schema_to_dlq():
    """v1 형식 오류는 INVALID_EVENT_SCHEMA로 격리합니다."""
    consumer, _, handler = create_consumer()
    message = create_message(
        body=json.dumps(
            {
                "schemaVersion": "1.0",
                "eventId": "not-a-uuid",
                "analysisId": 0,
            }
        ).encode("utf-8")
    )

    await consumer._on_message(message)

    handler.handle.assert_not_awaited()
    consumer.dead_letter_publisher.publish.assert_awaited_once_with(
        original_message_id=message.message_id,
        failure_code="INVALID_EVENT_SCHEMA",
        request_event=None,
    )
    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_requeues_first_processing_failure():
    """최초 분석 실패 재시도 테스트"""
    consumer, _, handler = create_consumer()
    message = create_message(redelivered=False)

    handler.handle.side_effect = RuntimeError(
        "Temporary analysis failure"
    )

    await consumer._on_message(message)

    handler.handle.assert_awaited_once()
    message.nack.assert_awaited_once_with(
        requeue=True
    )
    message.ack.assert_not_awaited()
    message.reject.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_does_not_treat_redelivery_as_retry_attempt():
    """Broker redelivery is not an application retry attempt."""
    consumer, _, handler = create_consumer()
    message = create_message(redelivered=True)

    handler.handle.side_effect = RuntimeError(
        "Temporary analysis failure"
    )

    await consumer._on_message(message)

    handler.handle.assert_awaited_once()
    message.nack.assert_awaited_once_with(
        requeue=True
    )
    message.ack.assert_not_awaited()
    message.reject.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_routes_to_dlq_after_retry_fails():
    """기록된 재시도까지 실패하면 정제 DLQ로 격리합니다."""
    consumer, _, handler = create_consumer()
    first_message = create_message(redelivered=False)
    second_message = create_message(redelivered=True)

    handler.handle.side_effect = RuntimeError(
        "Analysis failure"
    )

    await consumer._on_message(first_message)
    await consumer._on_message(second_message)

    first_message.nack.assert_awaited_once_with(
        requeue=True
    )
    dlq_arguments = (
        consumer.dead_letter_publisher
        .publish.call_args.kwargs
    )
    assert (
        dlq_arguments["original_message_id"]
        == second_message.message_id
    )
    assert (
        dlq_arguments["failure_code"]
        == "PROCESSING_RETRIES_EXHAUSTED"
    )
    assert dlq_arguments["request_event"].analysisId == 123
    second_message.ack.assert_awaited_once()
    second_message.reject.assert_not_awaited()
    assert consumer.retry_attempts == {}


@pytest.mark.asyncio
async def test_consumer_requeues_when_dlq_publication_fails():
    """DLQ 발행 실패 시 원본 유실을 막기 위해 requeue합니다."""
    consumer, _, handler = create_consumer()
    message = create_message(body=b"{invalid-json")

    consumer.dead_letter_publisher.publish.side_effect = (
        RuntimeError("DLQ unavailable")
    )

    await consumer._on_message(message)

    handler.handle.assert_not_awaited()
    message.ack.assert_not_awaited()
    message.nack.assert_awaited_once_with(requeue=True)
    message.reject.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumer_routes_non_retryable_processing_error_to_dlq():
    """처리 중 재시도 불가 오류는 즉시 DLQ로 격리합니다."""
    consumer, _, handler = create_consumer()
    message = create_message()

    handler.handle.side_effect = NonRetryableProcessingError(
        message="Invalid provider credentials",
        failure_code="INVALID_PROVIDER_CREDENTIALS",
    )

    await consumer._on_message(message)

    dlq_arguments = (
        consumer.dead_letter_publisher
        .publish.call_args.kwargs
    )
    assert (
        dlq_arguments["failure_code"]
        == "INVALID_PROVIDER_CREDENTIALS"
    )
    assert dlq_arguments["request_event"].analysisId == 123
    message.ack.assert_awaited_once()
    message.nack.assert_not_awaited()

@pytest.mark.asyncio
async def test_consumer_clears_retry_state_after_success():
    """Clear application retry state after successful processing."""
    consumer, _, handler = create_consumer()
    message = create_message()
    event_id = "1fb898fa-d89d-4d0b-a43f-a8b00daeb765"

    consumer.retry_attempts[event_id] = 1
    handler.handle.return_value = create_success_result()

    await consumer._on_message(message)

    assert event_id not in consumer.retry_attempts
    message.ack.assert_awaited_once()

@pytest.mark.asyncio
async def test_consumer_start_subscribes_to_queue():
    """Consumer 시작 테스트"""
    consumer, request_queue, _ = create_consumer()

    request_queue.consume.return_value = (
        "analysis-consumer-tag"
    )

    await consumer.start()

    request_queue.consume.assert_awaited_once_with(
        consumer._on_message,
        no_ack=False,
    )
    assert consumer.consumer_tag == (
        "analysis-consumer-tag"
    )

@pytest.mark.asyncio
async def test_consumer_start_is_idempotent():
    """Consumer 중복 시작 방지 테스트"""
    consumer, request_queue, _ = create_consumer()

    request_queue.consume.return_value = (
        "analysis-consumer-tag"
    )

    await consumer.start()
    await consumer.start()

    request_queue.consume.assert_awaited_once()

@pytest.mark.asyncio
async def test_consumer_stop_cancels_subscription():
    """Consumer 중단 테스트"""
    consumer, request_queue, _ = create_consumer()
    consumer.consumer_tag = "analysis-consumer-tag"

    await consumer.stop()

    request_queue.cancel.assert_awaited_once_with(
        "analysis-consumer-tag"
    )
    assert consumer.consumer_tag is None

@pytest.mark.asyncio
async def test_consumer_stop_before_start_does_nothing():
    consumer, request_queue, _ = create_consumer()

    await consumer.stop()

    request_queue.cancel.assert_not_awaited()
