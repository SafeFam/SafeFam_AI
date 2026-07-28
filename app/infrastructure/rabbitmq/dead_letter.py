import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractRobustExchange

from app.core.config import Settings, settings
from app.infrastructure.errors import (
    RetryableProcessingError,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
    DeadLetterEvent,
)

logger = logging.getLogger(__name__)


class DeadLetterPublisher:
    """개인정보가 제거된 실패 이벤트를 DLQ로 발행"""

    def __init__(
        self,
        exchange: AbstractRobustExchange,
        app_settings: Settings = settings,
    ) -> None:
        self.exchange = exchange
        self.settings = app_settings

    async def publish(
        self,
        *,
        original_message_id: str | None,
        failure_code: str,
        request_event: AnalysisRequestedEvent | None,
    ) -> DeadLetterEvent:
        """실패 사유와 기본 식별자 정보만 담은 DLQ 이벤트 생성 후 RabbitMQ로 발행"""

        # 요청 이벤트에서 비식별 추적 정보만 추출하여 DLQ 이벤트 생성
        dead_letter_event = DeadLetterEvent(
            schemaVersion="1.0",
            eventId=uuid4(),
            originalMessageId=original_message_id,
            analysisId=(
                request_event.analysisId
                if request_event is not None
                else None
            ),
            traceId=(
                request_event.traceId
                if request_event is not None
                else None
            ),
            failureCode=failure_code,
            failedAt=datetime.now(timezone.utc),
        )

        # DLQ 발행용 aio_pika 메시지 객체 생성
        message = Message(
            body=dead_letter_event.model_dump_json(
                by_alias=True
            ).encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=str(dead_letter_event.eventId),
            correlation_id=(
                str(dead_letter_event.traceId)
                if dead_letter_event.traceId is not None
                else None
            ),
            headers={
                "schemaVersion": (
                    dead_letter_event.schemaVersion
                ),
                "failureCode": failure_code,
                "sanitized": True,
            },
        )

        try:
            # DLQ 라우팅 키로 메시지 발행
            await asyncio.wait_for(
                self.exchange.publish(
                    message,
                    routing_key=(
                        self.settings
                        .RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY
                    ),
                    mandatory=True,
                ),
                timeout=(
                    self.settings
                    .RABBITMQ_PUBLISH_TIMEOUT_SECONDS
                ),
            )
        except TimeoutError as exception:
            raise RetryableProcessingError(
                message=(
                    "Sanitized dead-letter event "
                    "publication timed out"
                ),
                failure_code="DLQ_PUBLISH_TIMEOUT",
            ) from exception
        except Exception as exception:
            raise RetryableProcessingError(
                message=(
                    "Failed to publish sanitized "
                    "dead-letter event"
                ),
                failure_code="DLQ_PUBLISH_FAILED",
            ) from exception

        logger.warning(
            "Published sanitized dead-letter event. "
            "event_id=%s original_message_id=%s "
            "analysis_id=%s trace_id=%s failure_code=%s",
            dead_letter_event.eventId,
            original_message_id,
            dead_letter_event.analysisId,
            dead_letter_event.traceId,
            failure_code,
        )

        return dead_letter_event