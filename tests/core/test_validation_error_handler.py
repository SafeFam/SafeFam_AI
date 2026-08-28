from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.analysis.schemas import SmishingAnalysisRequest
from app.chat.schemas import ChatRequest
from app.core.config import settings
from app.core.exception_handlers import validation_exception_handler


def _build_client() -> TestClient:
    """실제 요청 스키마 + 실제 검증 핸들러를 배선한 최소 앱.

    AWS 등 서비스 의존 없이 422 응답 형태만 검증한다.
    """
    application = FastAPI()
    application.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    @application.post("/analyze")
    async def analyze(payload: SmishingAnalysisRequest) -> dict:
        return {"ok": True}

    @application.post("/chat")
    async def chat(payload: ChatRequest) -> dict:
        return {"ok": True}

    return TestClient(application)


def test_oversized_analyze_text_returns_422_without_leaking_pii():
    client = _build_client()
    secret = "01012345678-비밀번호-보이스피싱"

    response = client.post(
        "/analyze",
        json={"text": secret + "가" * settings.MAX_ANALYSIS_CONTENT_LENGTH},
    )

    assert response.status_code == 422
    assert secret not in response.text
    # 어떤 필드가 왜 거부됐는지는 여전히 알려준다.
    assert response.json()["detail"][0]["loc"][-1] == "text"


def test_oversized_chat_content_returns_422_without_leaking_pii():
    client = _build_client()
    secret = "01012345678-비밀번호"

    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": secret
                    + "가" * settings.MAX_CHAT_CONTENT_LENGTH,
                }
            ]
        },
    )

    assert response.status_code == 422
    assert secret not in response.text


def test_oversized_analysis_context_returns_422_without_leaking_pii():
    client = _build_client()
    secret = "01012345678-비밀번호"

    response = client.post(
        "/chat",
        json={
            "analysisContext": {
                "riskScore": 90,
                "riskLevel": "HIGH",
                "category": "FINANCIAL_INSTITUTION",
                "explanation": secret
                + "가" * settings.MAX_CHAT_CONTEXT_TEXT_LENGTH,
            },
            "messages": [{"role": "user", "content": "이거 진짜인가요?"}],
        },
    )

    assert response.status_code == 422
    assert secret not in response.text
