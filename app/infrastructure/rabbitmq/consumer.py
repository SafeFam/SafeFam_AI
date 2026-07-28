import logging

from aio_pika.abc import (
    AbstractIncomingMessage,
    AbstractRobustQueue,
)
from pydantic import ValidationError

from app.infrastructure.rabbitmq.handler import (
    AnalysisRequestHandler,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
)

logger = logging.getLogger(__name__)

class AnalysisRequestConsumer:
    """Spring의 분석 요청 이벤트를 수신하고 처리를 제어"""

    def __init__(
        self,
        request_queue: AbstractRobustQueue,
        handler: AnalysisRequestHandler,
    ) -> None:
        self.request_queue = request_queue
        self.handler = handler
        self.consumer_tag: str | None = None

    async def start(self) -> None:
        """Consumer 수신 시작"""
        if self.consumer_tag is not None:
            logger.info(
                "Analysis request consumer is already running. "
                "consumer_tag=%s",
                self.consumer_tag,
            )
            return

        # Consumer 시작
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
        """Consumer 수신 중단"""
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
        """메시지를 검증하고 분석 핸들러 호출"""
        try:
            event = AnalysisRequestedEvent.model_validate_json(
                message.body
            )
        except ValidationError:
            logger.exception(
                "Rejecting invalid analysis request event. "
                "message_id=%s",
                message.message_id,
            )

            await message.reject(requeue=False)
            return

        try:
            await self.handler.handle(event)
        except Exception:
            await self._handle_processing_failure(
                message=message,
                event=event,
            )
            return

        await message.ack()

        logger.info(
            "Analysis request acknowledged. "
            "message_id=%s event_id=%s "
            "analysis_id=%s trace_id=%s",
            message.message_id,
            event.eventId,
            event.analysisId,
            event.traceId,
        )

    async def _handle_processing_failure(
        self,
        message: AbstractIncomingMessage,
        event: AnalysisRequestedEvent,
    ) -> None:
        """분석 실패 시 1회 재시도 후 최종 Reject 처리"""
        if message.redelivered:
            logger.exception(
                "Analysis request failed after retry. "
                "Rejecting the message. "
                "message_id=%s event_id=%s "
                "analysis_id=%s trace_id=%s",
                message.message_id,
                event.eventId,
                event.analysisId,
                event.traceId,
            )

            await message.reject(requeue=False)
            return

        logger.exception(
            "Analysis request processing failed. "
            "Requeueing the message for one retry. "
            "message_id=%s event_id=%s "
            "analysis_id=%s trace_id=%s",
            message.message_id,
            event.eventId,
            event.analysisId,
            event.traceId,
        )

        await message.nack(requeue=True)