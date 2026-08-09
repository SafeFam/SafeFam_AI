import asyncio
import logging

from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractRobustExchange

from app.core.config import Settings, settings
from app.infrastructure.errors import (
    RetryableProcessingError,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisResultEvent,
)

logger = logging.getLogger(__name__)


class AnalysisResultPublisher:
    """RabbitMQ Exchange로 분석 결과 이벤트 발행"""

    def __init__(
        self,
        exchange: AbstractRobustExchange,
        app_settings: Settings = settings,
    ) -> None:
        self.exchange = exchange
        self.settings = app_settings

    async def publish(
        self,
        event: AnalysisResultEvent,
    ) -> None:
        """분석 결과 이벤트를 JSON 메시지로 직렬화하여 라우팅 키로 발행"""

        # 이벤트 타입(COMPLETED, PARTIAL, FAILED)에 따른 라우팅 키 바인딩
        routing_key = {
            AnalysisEventType.COMPLETED: self.settings.RABBITMQ_ANALYSIS_COMPLETED_ROUTING_KEY,
            AnalysisEventType.PARTIAL: self.settings.RABBITMQ_ANALYSIS_PARTIAL_ROUTING_KEY,
            AnalysisEventType.FAILED: self.settings.RABBITMQ_ANALYSIS_FAILED_ROUTING_KEY,
        }[event.eventType]

        # aio_pika 메시지 객체 생성
        message = Message(
            body=event.model_dump_json(by_alias=True).encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=str(event.eventId),
            correlation_id=str(event.traceId),
            headers={
                "schemaVersion": event.schemaVersion,
                "eventType": event.eventType.value,
            },
        )

        try:
            await asyncio.wait_for(
                self.exchange.publish(
                    message,
                    routing_key=routing_key,
                    mandatory=True,
                ),
                timeout=(self.settings.RABBITMQ_PUBLISH_TIMEOUT_SECONDS),
            )
        except TimeoutError as exception:
            raise RetryableProcessingError(
                message=("Analysis result publication timed out"),
                failure_code="RESULT_PUBLISH_TIMEOUT",
            ) from exception
        except Exception as exception:
            raise RetryableProcessingError(
                message=("Failed to publish analysis result"),
                failure_code="RESULT_PUBLISH_FAILED",
            ) from exception
