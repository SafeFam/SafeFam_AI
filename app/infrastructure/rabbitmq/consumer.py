import json
import logging
from json import JSONDecodeError

from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustQueue,
)
from pydantic import ValidationError

from app.analysis.execution import classify_execution
from app.infrastructure.errors import (
    NonRetryableProcessingError,
)
from app.infrastructure.rabbitmq.dead_letter import (
    DeadLetterPublisher,
)
from app.infrastructure.rabbitmq.handler import (
    AnalysisRequestHandler,
)
from app.infrastructure.rabbitmq.publisher import (
    AnalysisResultPublisher,
)
from app.infrastructure.rabbitmq.result_factory import (
    AnalysisResultEventFactory,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
)

logger = logging.getLogger(__name__)


class AnalysisRequestConsumer:
    """Spring 분석 요청을 처리하고 결과 이벤트 발행"""

    MAX_RETRY_ATTEMPTS = 1

    def __init__(
        self,
        request_queue: AbstractRobustQueue,
        handler: AnalysisRequestHandler,
        result_publisher: AnalysisResultPublisher,
        result_factory: AnalysisResultEventFactory,
        dead_letter_publisher: DeadLetterPublisher,
    ) -> None:
        self.request_queue = request_queue
        self.handler = handler
        self.result_publisher = result_publisher
        self.result_factory = result_factory
        self.dead_letter_publisher = dead_letter_publisher

        self.consumer_tag: str | None = None
        self.retry_attempts: dict[str, int] = {}

    async def start(self) -> None:
        """분석 요청 Consumer 시작"""
        if self.consumer_tag is not None:
            logger.info(
                "Analysis request consumer is already running. "
                "consumer_tag=%s",
                self.consumer_tag,
            )
            return

        self.consumer_tag = await self.request_queue.consume(
            self._on_message,
            no_ack=False,
        )

        logger.info(
            "Analysis request consumer started. "
            "consumer_tag=%s",
            self.consumer_tag,
        )

    async def stop(self) -> None:
        """분석 요청 Consumer 구독 중단"""
        if self.consumer_tag is None:
            return

        consumer_tag = self.consumer_tag

        await self.request_queue.cancel(consumer_tag)
        self.consumer_tag = None

        logger.info(
            "Analysis request consumer stopped. "
            "consumer_tag=%s",
            consumer_tag,
        )

    async def _on_message(
        self,
        message: AbstractIncomingMessage,
    ) -> None:
        """요청 처리 및 결과 발행 성공 후 ACK"""
        try:
            event = self._parse_event(message.body)
        except NonRetryableProcessingError as exception:
            await self._route_to_dead_letter(
                message=message,
                event=None,
                failure_code=exception.failure_code,
            )
            return

        try:
            result = await self.handler.handle(event)

            execution = classify_execution(result)

            result_event = self.result_factory.create(
                request=event,
                execution=execution,
            )

            await self.result_publisher.publish(
                result_event
            )
        except NonRetryableProcessingError as exception:
            await self._route_to_dead_letter(
                message=message,
                event=event,
                failure_code=exception.failure_code,
            )
            return
        except Exception as exception:
            await self._handle_retryable_failure(
                message=message,
                event=event,
                exception=exception,
            )
            return

        self.retry_attempts.pop(
            str(event.eventId),
            None,
        )

        await message.ack()

        logger.info(
            "Analysis request acknowledged after result publication. "
            "message_id=%s event_id=%s "
            "analysis_id=%s trace_id=%s "
            "result_event_id=%s result_event_type=%s",
            message.message_id,
            event.eventId,
            event.analysisId,
            event.traceId,
            result_event.eventId,
            result_event.eventType.value,
        )

    def _parse_event(
        self,
        body: bytes,
    ) -> AnalysisRequestedEvent:
        """JSON과 이벤트 계약을 단계적으로 검증"""
        try:
            raw_event = json.loads(body)
        except (JSONDecodeError, UnicodeDecodeError) as exception:
            raise NonRetryableProcessingError(
                message="Invalid JSON analysis request",
                failure_code="INVALID_JSON",
            ) from exception

        if not isinstance(raw_event, dict):
            raise NonRetryableProcessingError(
                message="Analysis event must be an object",
                failure_code="INVALID_EVENT_SCHEMA",
            )

        if raw_event.get("schemaVersion") != "1.0":
            raise NonRetryableProcessingError(
                message="Unsupported schema version",
                failure_code="UNSUPPORTED_SCHEMA_VERSION",
            )

        try:
            return AnalysisRequestedEvent.model_validate(
                raw_event
            )
        except ValidationError as exception:
            raise NonRetryableProcessingError(
                message="Invalid analysis event schema",
                failure_code="INVALID_EVENT_SCHEMA",
            ) from exception

    async def _handle_retryable_failure(
        self,
        message: AbstractIncomingMessage,
        event: AnalysisRequestedEvent,
        exception: Exception,
    ) -> None:
        """일시적 오류를 1회 재시도한 뒤 DLQ로 격리"""
        event_id = str(event.eventId)
        retry_attempt = self.retry_attempts.get(
            event_id,
            0,
        )

        if retry_attempt >= self.MAX_RETRY_ATTEMPTS:
            self.retry_attempts.pop(event_id, None)

            logger.error(
                "Analysis request failed after retry. "
                "Routing sanitized event to DLQ. "
                "message_id=%s event_id=%s "
                "analysis_id=%s trace_id=%s "
                "error_type=%s",
                message.message_id,
                event.eventId,
                event.analysisId,
                event.traceId,
                exception.__class__.__name__,
            )

            await self._route_to_dead_letter(
                message=message,
                event=event,
                failure_code=(
                    "PROCESSING_RETRIES_EXHAUSTED"
                ),
            )
            return

        self.retry_attempts[event_id] = (
            retry_attempt + 1
        )

        logger.warning(
            "Analysis request processing failed. "
            "Requeueing for retry. "
            "message_id=%s event_id=%s "
            "analysis_id=%s trace_id=%s "
            "retry_attempt=%s error_type=%s",
            message.message_id,
            event.eventId,
            event.analysisId,
            event.traceId,
            retry_attempt + 1,
            exception.__class__.__name__,
        )

        await message.nack(requeue=True)

    async def _route_to_dead_letter(
        self,
        *,
        message: AbstractIncomingMessage,
        event: AnalysisRequestedEvent | None,
        failure_code: str,
    ) -> None:
        """정제된 DLQ 이벤트 발행 성공 후 원본을 ACK"""
        try:
            await self.dead_letter_publisher.publish(
                original_message_id=message.message_id,
                failure_code=failure_code,
                request_event=event,
            )
        except Exception as exception:
            logger.error(
                "Failed to publish sanitized DLQ event. "
                "message_id=%s failure_code=%s "
                "error_type=%s",
                message.message_id,
                failure_code,
                exception.__class__.__name__,
            )

            await message.nack(requeue=True)
            return

        if event is not None:
            self.retry_attempts.pop(
                str(event.eventId),
                None,
            )

        await message.ack()

        logger.warning(
            "Original analysis request acknowledged "
            "after sanitized DLQ publication. "
            "message_id=%s analysis_id=%s "
            "trace_id=%s failure_code=%s",
            message.message_id,
            event.analysisId if event else None,
            event.traceId if event else None,
            failure_code,
        )