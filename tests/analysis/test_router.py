"""POST /api/analyze 계약을 실제 HTTP 레벨에서 검증하는 테스트.

Stacking 결과는 시나리오별 고정값으로 대체하고 Gemini와 URL 공급자는
mock 모드로 실행해 외부 API 및 학습 artifact 상태에 의존하지 않는다.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

app = create_app(rabbitmq_consumer_enabled=False)
client = TestClient(app)


def _stacking_result(text: str) -> dict:
    """일상 문장은 확실한 정상, 그 외 문장은 불확실 구간으로 반환."""

    probability = 0.1 if (
        "소주" in text or "테스트 메시지" in text
    ) else 0.5

    return {
        "engine": "stacking",
        "is_available": True,
        "result": {
            "risk_score": round(probability * 100),
            "risk_probability": probability,
            "confidence": 0.8,
            "is_suspected_phishing": probability >= 0.5,
            "threshold": 0.2,
            "model_scores": {},
            "unavailable_models": [],
            "error_message": None,
        },
    }


@pytest.fixture(autouse=True)
def _mock_external_paid_apis():
    with (
        patch("app.analysis.text.llm_analyzer.MOCK_ENABLED", True),
        patch("app.analysis.url.analyzer.MOCK_ENABLED", True),
        patch(
            "app.analysis.service.analyze_text_with_stacking",
            side_effect=_stacking_result,
        ),
        patch(
            "app.analysis.service.settings."
            "STACKING_NORMAL_PROBABILITY_MAX",
            0.2,
        ),
        patch(
            "app.analysis.service.settings."
            "STACKING_PHISHING_PROBABILITY_MIN",
            0.8,
        ),
    ):
        yield


def test_casual_message_is_low_risk_and_skips_gemini():
    """
    시나리오: 일상 대화 문자를 stacking이 확실한 정상으로 판정하면
    Gemini를 호출하지 않고 LOW 등급으로 응답해야 한다.
    """
    response = client.post("/api/analyze", json={"text": "오늘 소주 한잔 고?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["risk_grade"] == "LOW"
    assert body["url_analysis"] is None
    assert body["text_analysis"]["engine"] == "hybrid_stacking_llm"
    assert body["text_analysis"]["llm_called"] is False
    assert body["text_analysis"]["decision_source"] == "STACKING"
    assert body["text_analysis"]["self_model"]["risk_score"] == 10


def test_phishing_text_without_url_escalates_to_gemini_and_is_high_risk():
    """
    시나리오: URL 없이 기관 사칭 + 긴급성 유도 문구만 있는 전형적 스미싱 문자 ->
    stacking이 불확실 구간으로 판정해 Gemini 재검증을 실행하고,
    최종 위험 등급도 높게 나와야 한다.
    """
    text = "[검찰청] 귀하 명의로 대포통장이 개설되어 수사가 진행 중입니다. 즉시 아래 링크로 접속하여 신원을 확인하세요."
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["text_analysis"]["llm_called"] is True
    assert body["text_analysis"]["llm_available"] is True
    assert body["text_analysis"]["decision_source"] == "LLM"
    assert body["text_analysis"]["self_model"]["risk_score"] == 50
    assert body["risk_grade"] in ("MEDIUM", "HIGH")
    assert body["url_analysis"] is None


def test_message_with_url_populates_url_analysis_via_hybrid_engine():
    """
    시나리오: URL이 포함된 문자 -> URL 트랙이 실행되어 url_analysis가 채워지고,
    Mock 하이브리드 엔진의 악성 판정이 최종 스코어의 hybrid_url 기여분에 반영되어야 한다.
    trace_url 자체는 이 테스트의 관심사가 아니므로(리다이렉트 동작은 전담 URL 트레이서
    테스트에서 검증) 리다이렉트 없이 원본 URL을 그대로 돌려주도록 Mock 처리한다.
    """
    text = "국민은행 보안 업데이트 안내입니다. https://www.google.com 확인해주세요."

    with patch("app.analysis.service.trace_url", new_callable=AsyncMock) as mock_trace:
        mock_trace.return_value = "https://www.google.com"
        response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["url_analysis"]["has_url"] is True
    assert body["url_analysis"]["original_url"] == "https://www.google.com"
    assert body["url_analysis"]["engine_source"] == "Hybrid-Engine (MOCK)"
    assert body["url_analysis"]["is_url_malicious"] is True
    assert body["contribution_breakdown"]["hybrid_url"] > 0


def test_response_matches_swagger_documented_schema_contract():
    """Swagger가 문서화하는 SmishingAnalysisResponse의 필드/값 범위가 항상 지켜지는지 검증."""
    response = client.post("/api/analyze", json={"text": "테스트 메시지입니다"})

    assert response.status_code == 200
    body = response.json()

    assert {
        "status",
        "message",
        "final_score",
        "risk_grade",
        "contribution_breakdown",
        "text_analysis",
        "url_analysis",
    }.issubset(body.keys())

    assert 0 <= body["final_score"] <= 100
    assert body["risk_grade"] in ("HIGH", "MEDIUM", "LOW")

    # URL 없으면 그 30%가 LLM/규칙 트랙으로 재배분되어 상한이 65/35로 늘어남
    breakdown = body["contribution_breakdown"]
    assert 0 <= breakdown["llm"] <= 65
    assert 0 <= breakdown["hybrid_url"] <= 30
    assert 0 <= breakdown["rules"] <= 35


def test_rejects_request_missing_required_text_field():
    """Swagger 계약상 필수 필드(text)가 없으면 FastAPI가 422 Validation Error로 응답해야 한다."""
    response = client.post("/api/analyze", json={})
    assert response.status_code == 422


def test_rejects_request_with_wrong_type_for_text_field():
    response = client.post("/api/analyze", json={"text": 12345})
    assert response.status_code == 422


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_openapi_schema_documents_analyze_endpoint():
    """Swagger UI(/docs)가 실제로 참조하는 OpenAPI 스키마에 /api/analyze 계약이 노출되는지 검증."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/analyze" in schema["paths"]
    assert "post" in schema["paths"]["/api/analyze"]
