from unittest.mock import AsyncMock, MagicMock, patch

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

    fake_consumer = MagicMock()
    fake_consumer.start = AsyncMock()
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
        with TestClient(application) as test_client:
            response = test_client.get("/")

            assert response.status_code == 200
            fake_connection.connect.assert_awaited_once()
            fake_consumer.start.assert_awaited_once()

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