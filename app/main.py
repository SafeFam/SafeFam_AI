from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.analysis import router as analyze
from app.analysis.service import SmishingAnalysisService
from app.chat import router as chat
from app.core.config import settings
from app.infrastructure.rabbitmq.connection import (
    RabbitMQConnection,
)
from app.infrastructure.rabbitmq.consumer import (
    AnalysisRequestConsumer,
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
from app.infrastructure.rabbitmq.dead_letter import (
    DeadLetterPublisher,
)


def create_lifespan(
    rabbitmq_consumer_enabled: bool,
):
    """FastAPI 애플리케이션 시작 및 종료 시 RabbitMQ 리소스 생명주기를 관리"""
    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        if not rabbitmq_consumer_enabled:
            yield
            return

        rabbitmq = RabbitMQConnection()
        consumer: AnalysisRequestConsumer | None = None

        try:
            await rabbitmq.connect()

            handler = AnalysisRequestHandler(
                analysis_service=SmishingAnalysisService()
            )

            result_publisher = AnalysisResultPublisher(
                exchange=rabbitmq.get_exchange(),
            )

            result_factory = AnalysisResultEventFactory()

            dead_letter_publisher = DeadLetterPublisher(
                exchange=rabbitmq.get_exchange(),
            )

            consumer = AnalysisRequestConsumer(
                request_queue=rabbitmq.get_request_queue(),
                handler=handler,
                result_publisher=result_publisher,
                result_factory=result_factory,
                dead_letter_publisher=dead_letter_publisher,
            )

            await consumer.start()

            application.state.rabbitmq = rabbitmq
            application.state.analysis_request_consumer = (
                consumer
            )

            yield
        finally:
            try:
                if consumer is not None:
                    await consumer.stop()
            finally:
                await rabbitmq.close()

    return lifespan


def create_app(
    *,
    rabbitmq_consumer_enabled: bool | None = None,
) -> FastAPI:
    """FastAPI 애플리케이션 인스턴스를 생성하고 미들웨어, 라우터 및 Lifespan을 설정"""
    consumer_enabled = (
        settings.RABBITMQ_CONSUMER_ENABLED
        if rabbitmq_consumer_enabled is None
        else rabbitmq_consumer_enabled
    )

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=create_lifespan(consumer_enabled),
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(analyze.router, prefix="/api")
    application.include_router(chat.router, prefix="/api")

    @application.get("/", tags=["Root"])
    def root_check():
        return {
            "status": "healthy",
            "project": settings.PROJECT_NAME,
        }

    return application


app = create_app()
