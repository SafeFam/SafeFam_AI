"""
Swagger UI(/docs)가 문서화하는 POST /api/analyze 계약을 실제 HTTP 레벨에서 검증하는 E2E 시나리오 테스트.

- 나이브 베이즈(1차)는 실제 로직을 그대로 태운다 (로컬/무료 자원).
- URL 리다이렉트 추적기(trace_url)는 리다이렉트가 없는 고정값으로 Mock 처리한다 -
  실제 동작은 tests/url/test_url_tracer.py에서 전담 검증하며, 여기서는 outbound
  DNS/HTTP 호출과 CI 아웃바운드 의존성을 없애기 위함이다.
- Gemini(2차)와 GSB+VT 하이브리드 URL 엔진은 결정론적인 Mock 모드로 우회한다 (유료/외부 API 의존성 제거).
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _mock_external_paid_apis():
    with patch("app.analysis.text.gemini_analyzer.MOCK_ENABLED", True), \
         patch("app.analysis.url.analyzer.MOCK_ENABLED", True):
        yield


def test_casual_message_is_low_risk_and_skips_gemini():
    """
    시나리오: 일상 대화 문자 -> 나이브 베이즈가 SAFE로 판정해 Gemini 2차 검증을 스킵하고
    LOW 등급으로 응답해야 한다.
    """
    response = client.post("/api/analyze", json={"text": "오늘 소주 한잔 고?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["risk_grade"] == "LOW"
    assert body["url_analysis"] is None
    assert body["text_analysis"]["engine"] == "naive_bayes"
    assert "stage1_naive_bayes" not in body["text_analysis"]


def test_phishing_text_without_url_escalates_to_gemini_and_is_high_risk():
    """
    시나리오: URL 없이 기관 사칭 + 긴급성 유도 문구만 있는 전형적 스미싱 문자 ->
    나이브 베이즈가 의심 판정해 Gemini 2차 검증까지 실행되고, 최종 위험 등급도 높게 나와야 한다.
    """
    text = "[검찰청] 귀하 명의로 대포통장이 개설되어 수사가 진행 중입니다. 즉시 아래 링크로 접속하여 신원을 확인하세요."
    response = client.post("/api/analyze", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert "stage1_naive_bayes" in body["text_analysis"]
    assert body["text_analysis"]["stage1_naive_bayes"]["risk_score"] == 97
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
        "status", "message", "final_score", "risk_grade",
        "contribution_breakdown", "text_analysis", "url_analysis"
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
