import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app

app = create_app(rabbitmq_consumer_enabled=False)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _mock_gemini_chat():
    with patch("app.chat.service.MOCK_ENABLED", True):
        yield


def _payload(**overrides) -> dict:
    payload = {
        "analysisContext": {
            "riskScore": 90,
            "riskLevel": "HIGH",
            "category": "FINANCIAL_INSTITUTION",
            "explanation": "국민건강보험을 사칭한 스미싱 문자",
            "indicators": [
                {"type": "URGENCY_KEYWORD", "description": "즉시 확인 유도"}
            ],
        },
        "messages": [{"role": "user", "content": "이거 진짜인가요?"}],
    }
    payload.update(overrides)
    return payload


def test_chat_endpoint_returns_message():
    response = client.post("/api/chat", json=_payload())

    assert response.status_code == 200
    assert "message" in response.json()


def test_chat_endpoint_rejects_empty_message_history():
    response = client.post("/api/chat", json=_payload(messages=[]))
    assert response.status_code == 422


def test_chat_endpoint_rejects_invalid_role():
    response = client.post(
        "/api/chat",
        json=_payload(messages=[{"role": "system", "content": "hi"}]),
    )
    assert response.status_code == 422


def test_chat_endpoint_rejects_blank_message_content():
    response = client.post(
        "/api/chat",
        json=_payload(messages=[{"role": "user", "content": "   "}]),
    )
    assert response.status_code == 422


def test_chat_endpoint_rejects_missing_analysis_context():
    payload = _payload()
    del payload["analysisContext"]
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422


def test_chat_endpoint_returns_502_on_service_error():
    with patch("app.chat.service.MOCK_ENABLED", False), \
         patch("app.chat.service.GEMINI_API_KEY", None):
        response = client.post("/api/chat", json=_payload())

    assert response.status_code == 502


def test_openapi_schema_documents_chat_endpoint():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/chat" in schema["paths"]
    assert "post" in schema["paths"]["/api/chat"]
