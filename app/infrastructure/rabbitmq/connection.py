import logging

import aio_pika
from aio_pika import ExchangeType
from aio_pika.abc import (
    AbstractRobustChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
    AbstractRobustQueue,
)

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)

class RabbitMQNotConnectedError(RuntimeError):
    """RabbitMQ 연결 초기화 전에 리소스에 접근한 경우"""

class RabbitMQConnection:
    """RabbitMQ 연결과 분석 요청 토폴로지 관리"""

    def __init__(
        self,
        app_settings: Settings = settings,
    ) -> None:
        self.settings = app_settings

        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractRobustChannel | None = None
        self.exchange: AbstractRobustExchange | None = None
        self.request_queue: AbstractRobustQueue | None = None

    async def connect(self) -> None:
        """RabbitMQ 연결 및 Exchange, Queue, Binding 초기화"""
        if (
            self.connection is not None
            and not self.connection.is_closed
        ):
            logger.info("RabbitMQ connection is already active.")
            return

        logger.info("Connecting to RabbitMQ.")

        self.connection = await aio_pika.connect_robust(
            self.settings.RABBITMQ_URL
        )

        self.channel = await self.connection.channel()

        await self.channel.set_qos(
            prefetch_count=(
                self.settings.RABBITMQ_PREFETCH_COUNT
            )
        )

        self.exchange = await self.channel.declare_exchange(
            self.settings.RABBITMQ_ANALYSIS_EXCHANGE,
            ExchangeType.TOPIC,
            durable=True,
        )

        self.request_queue = await self.channel.declare_queue(
            self.settings.RABBITMQ_ANALYSIS_REQUEST_QUEUE,
            durable=True,
        )

        await self.request_queue.bind(
            self.exchange,
            routing_key=(
                self.settings
                .RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY
            ),
        )

        logger.info(
            "RabbitMQ analysis request topology initialized. "
            "exchange=%s queue=%s routing_key=%s prefetch=%s",
            self.settings.RABBITMQ_ANALYSIS_EXCHANGE,
            self.settings.RABBITMQ_ANALYSIS_REQUEST_QUEUE,
            self.settings.RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY,
            self.settings.RABBITMQ_PREFETCH_COUNT,
        )

    def get_request_queue(self) -> AbstractRobustQueue:
        """초기화된 분석 요청 큐 객체 반환"""
        if self.request_queue is None:
            raise RabbitMQNotConnectedError(
                "RabbitMQ request queue is not initialized."
            )

        return self.request_queue

    async def close(self) -> None:
        if self.connection is None:
            return

        if not self.connection.is_closed:
            logger.info("Closing RabbitMQ connection.")
            await self.connection.close()

        self.connection = None
        self.channel = None
        self.exchange = None
        self.request_queue = None