from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.infrastructure.errors import (
    RetryableProcessingError,
)
from app.infrastructure.rabbitmq.publisher import (
    AnalysisResultPublisher,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
)


def _publisher():
    exchange = AsyncMock()
    app_settings = SimpleNamespace(
        RABBITMQ_ANALYSIS_COMPLETED_ROUTING_KEY=("analysis.completed.v1"),
        RABBITMQ_ANALYSIS_PARTIAL_ROUTING_KEY=("analysis.partial.v1"),
        RABBITMQ_ANALYSIS_FAILED_ROUTING_KEY=("analysis.failed.v1"),
        RABBITMQ_PUBLISH_TIMEOUT_SECONDS=1,
    )
    return (
        AnalysisResultPublisher(
            exchange,
            app_settings,
        ),
        exchange,
    )


def _event():
    event = Mock()
    event.eventType = AnalysisEventType.COMPLETED
    event.eventId = "result-event-id"
    event.traceId = "trace-id"
    event.schemaVersion = "1.0"
    event.model_dump_json.return_value = "{}"
    return event


@pytest.mark.asyncio
async def test_publisher_classifies_timeout_as_retryable():
    publisher, exchange = _publisher()
    exchange.publish.side_effect = TimeoutError

    with pytest.raises(
        RetryableProcessingError,
        match="timed out",
    ) as captured:
        await publisher.publish(_event())

    assert captured.value.failure_code == "RESULT_PUBLISH_TIMEOUT"


@pytest.mark.asyncio
async def test_publisher_classifies_publish_failure_as_retryable():
    publisher, exchange = _publisher()
    exchange.publish.side_effect = RuntimeError("unroutable")

    with pytest.raises(
        RetryableProcessingError,
        match="Failed to publish",
    ) as captured:
        await publisher.publish(_event())

    assert captured.value.failure_code == "RESULT_PUBLISH_FAILED"
