from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aio_pika import ExchangeType

from app.infrastructure.rabbitmq.connection import (
    RabbitMQConnection,
    RabbitMQNotConnectedError,
)


def create_fake_settings():
    return SimpleNamespace(
        RABBITMQ_URL=("amqp://test-user:test-password@localhost:5672/"),
        RABBITMQ_ANALYSIS_EXCHANGE="safefam.analysis",
        RABBITMQ_ANALYSIS_REQUEST_QUEUE=("safefam.analysis.requested.q"),
        RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY=("analysis.requested.v1"),
        RABBITMQ_ANALYSIS_DLQ=("safefam.analysis.requested.dlq"),
        RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY=("analysis.requested.dead.v1"),
        RABBITMQ_PREFETCH_COUNT=1,
    )


@pytest.mark.asyncio
@patch(
    "app.infrastructure.rabbitmq.connection.aio_pika.connect_robust",
    new_callable=AsyncMock,
)
async def test_connect_initializes_request_topology(
    mock_connect_robust: AsyncMock,
):
    """연결과 토폴로지 선언 테스트."""
    fake_connection = AsyncMock()
    fake_connection.is_closed = False

    fake_channel = AsyncMock()
    fake_exchange = AsyncMock()
    fake_queue = AsyncMock()
    fake_dead_letter_queue = AsyncMock()

    fake_connection.channel.return_value = fake_channel
    fake_channel.declare_exchange.return_value = fake_exchange
    fake_channel.declare_queue.side_effect = [
        fake_queue,
        fake_dead_letter_queue,
    ]

    mock_connect_robust.return_value = fake_connection

    app_settings = create_fake_settings()
    rabbitmq = RabbitMQConnection(app_settings)

    await rabbitmq.connect()

    mock_connect_robust.assert_awaited_once_with(app_settings.RABBITMQ_URL)
    fake_connection.channel.assert_awaited_once_with(
        publisher_confirms=True,
        on_return_raises=True,
    )

    fake_channel.set_qos.assert_awaited_once_with(prefetch_count=1)

    fake_channel.declare_exchange.assert_awaited_once_with(
        "safefam.analysis",
        ExchangeType.TOPIC,
        durable=True,
    )

    fake_channel.declare_queue.assert_any_await(
        "safefam.analysis.requested.q",
        durable=True,
    )
    fake_channel.declare_queue.assert_any_await(
        "safefam.analysis.requested.dlq",
        durable=True,
    )
    assert fake_channel.declare_queue.await_count == 2

    fake_queue.bind.assert_awaited_once_with(
        fake_exchange,
        routing_key="analysis.requested.v1",
    )
    fake_dead_letter_queue.bind.assert_awaited_once_with(
        fake_exchange,
        routing_key="analysis.requested.dead.v1",
    )

    assert rabbitmq.exchange is fake_exchange
    assert rabbitmq.request_queue is fake_queue
    assert rabbitmq.dead_letter_queue is fake_dead_letter_queue
    assert rabbitmq.get_exchange() is fake_exchange


@pytest.mark.asyncio
@patch(
    "app.infrastructure.rabbitmq.connection.aio_pika.connect_robust",
    new_callable=AsyncMock,
)
async def test_connect_does_not_open_duplicate_connection(
    mock_connect_robust: AsyncMock,
):
    """중복 연결 방지 테스트"""
    fake_connection = AsyncMock()
    fake_connection.is_closed = False

    fake_channel = AsyncMock()
    fake_exchange = AsyncMock()
    fake_queue = AsyncMock()

    fake_connection.channel.return_value = fake_channel
    fake_channel.declare_exchange.return_value = fake_exchange
    fake_channel.declare_queue.return_value = fake_queue
    mock_connect_robust.return_value = fake_connection

    rabbitmq = RabbitMQConnection(create_fake_settings())

    await rabbitmq.connect()
    await rabbitmq.connect()

    mock_connect_robust.assert_awaited_once()


def test_get_request_queue_fails_before_connect():
    """연결 전 Queue 접근 테스트"""
    rabbitmq = RabbitMQConnection(create_fake_settings())

    with pytest.raises(
        RabbitMQNotConnectedError,
        match="request queue is not initialized",
    ):
        rabbitmq.get_request_queue()


def test_get_exchange_fails_before_connect():
    """연결 전 Exchange 접근 테스트."""
    rabbitmq = RabbitMQConnection(create_fake_settings())

    with pytest.raises(
        RabbitMQNotConnectedError,
        match="exchange is not initialized",
    ):
        rabbitmq.get_exchange()


@pytest.mark.asyncio
async def test_close_closes_connection_and_clears_resources():
    """연결 종료 테스트"""
    rabbitmq = RabbitMQConnection(create_fake_settings())

    fake_connection = AsyncMock()
    fake_connection.is_closed = False

    rabbitmq.connection = fake_connection
    rabbitmq.channel = AsyncMock()
    rabbitmq.exchange = AsyncMock()
    rabbitmq.request_queue = AsyncMock()
    rabbitmq.dead_letter_queue = AsyncMock()

    await rabbitmq.close()

    fake_connection.close.assert_awaited_once()

    assert rabbitmq.connection is None
    assert rabbitmq.channel is None
    assert rabbitmq.exchange is None
    assert rabbitmq.request_queue is None
    assert rabbitmq.dead_letter_queue is None


@pytest.mark.asyncio
async def test_close_does_not_close_already_closed_connection():
    """이미 닫힌 연결 테스트"""
    rabbitmq = RabbitMQConnection(create_fake_settings())

    fake_connection = AsyncMock()
    fake_connection.is_closed = True
    rabbitmq.connection = fake_connection

    await rabbitmq.close()

    fake_connection.close.assert_not_awaited()
    assert rabbitmq.connection is None
