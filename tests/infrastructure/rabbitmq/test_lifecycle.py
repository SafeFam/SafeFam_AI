from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

def test_disabled_consumer_does_not_connect_to_rabbitmq():
    """Consumer가 비활성화됐을 때 RabbitMQ를 생성하지 않는지 테스트"""
    application = create_app(
        rabbitmq_consumer_enabled=False
    )

    with patch(
        "app.main.RabbitMQConnection"
    ) as mock_connection_class:
        with TestClient(application) as test_client:
            response = test_client.get("/")

    assert response.status_code == 200
    mock_connection_class.assert_not_called()

def test_lifespan_starts_and_stops_rabbitmq_consumer():
    """Consumer 시작/종료 Lifespan 테스트"""
    fake_connection = MagicMock()
    fake_connection.connect = AsyncMock()
    fake_connection.close = AsyncMock()

    fake_queue = MagicMock()
    fake_connection.get_request_queue.return_value = (
        fake_queue
    )
    fake_exchange = MagicMock()
    fake_connection.get_exchange.return_value = (
        fake_exchange
    )

    fake_consumer = MagicMock()
    fake_consumer.start = AsyncMock()
    fake_consumer.stop = AsyncMock()
    fake_result_publisher = MagicMock()
    fake_result_factory = MagicMock()
    fake_dead_letter_publisher = MagicMock()

    application = create_app(
        rabbitmq_consumer_enabled=True
    )

    with (
        patch(
            "app.main.RabbitMQConnection",
            return_value=fake_connection,
        ),
        patch(
            "app.main.AnalysisRequestConsumer",
            return_value=fake_consumer,
        ) as mock_consumer_class,
        patch(
            "app.main.AnalysisResultPublisher",
            return_value=fake_result_publisher,
        ) as mock_result_publisher_class,
        patch(
            "app.main.AnalysisResultEventFactory",
            return_value=fake_result_factory,
        ),
        patch(
            "app.main.DeadLetterPublisher",
            return_value=fake_dead_letter_publisher,
        ) as mock_dead_letter_publisher_class,
    ):
        with TestClient(application) as test_client:
            response = test_client.get("/")

            assert response.status_code == 200
            fake_connection.connect.assert_awaited_once()
            fake_consumer.start.assert_awaited_once()

        mock_result_publisher_class.assert_called_once_with(
            exchange=fake_exchange,
        )
        mock_dead_letter_publisher_class.assert_called_once_with(
            exchange=fake_exchange,
        )
        consumer_arguments = (
            mock_consumer_class.call_args.kwargs
        )
        assert (
            consumer_arguments["request_queue"]
            is fake_queue
        )
        assert (
            consumer_arguments["result_publisher"]
            is fake_result_publisher
        )
        assert (
            consumer_arguments["result_factory"]
            is fake_result_factory
        )
        assert (
            consumer_arguments["dead_letter_publisher"]
            is fake_dead_letter_publisher
        )

        fake_consumer.stop.assert_awaited_once()
        fake_connection.close.assert_awaited_once()

# Consumer 시작 실패 시 연결 정리 테스트
def test_lifespan_closes_connection_when_consumer_start_fails():
    fake_connection = MagicMock()
    fake_connection.connect = AsyncMock()
    fake_connection.close = AsyncMock()
    fake_connection.get_request_queue.return_value = (
        MagicMock()
    )

    fake_consumer = MagicMock()
    fake_consumer.start = AsyncMock(
        side_effect=RuntimeError("Consumer start failed")
    )
    fake_consumer.stop = AsyncMock()

    application = create_app(
        rabbitmq_consumer_enabled=True
    )

    with (
        patch(
            "app.main.RabbitMQConnection",
            return_value=fake_connection,
        ),
        patch(
            "app.main.AnalysisRequestConsumer",
            return_value=fake_consumer,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Consumer start failed",
        ):
            with TestClient(application):
                pass

    fake_connection.close.assert_awaited_once()


def test_lifespan_closes_connection_when_consumer_stop_fails():
    """Always close RabbitMQ when consumer shutdown fails."""
    fake_connection = MagicMock()
    fake_connection.connect = AsyncMock()
    fake_connection.close = AsyncMock()
    fake_connection.get_request_queue.return_value = (
        MagicMock()
    )

    fake_consumer = MagicMock()
    fake_consumer.start = AsyncMock()
    fake_consumer.stop = AsyncMock(
        side_effect=RuntimeError("Consumer stop failed")
    )

    application = create_app(
        rabbitmq_consumer_enabled=True
    )

    with (
        patch(
            "app.main.RabbitMQConnection",
            return_value=fake_connection,
        ),
        patch(
            "app.main.AnalysisRequestConsumer",
            return_value=fake_consumer,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Consumer stop failed",
        ):
            with TestClient(application):
                pass

    fake_consumer.stop.assert_awaited_once()
    fake_connection.close.assert_awaited_once()
