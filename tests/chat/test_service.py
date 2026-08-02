import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.chat.schemas import AnalysisContext, ChatMessage, ChatRequest, ChatRole
from app.chat.service import ChatService, ChatServiceError


def _request(messages=None) -> ChatRequest:
    return ChatRequest(
        analysisContext=AnalysisContext(
            riskScore=90,
            riskGrade="HIGH",
            phishingType="기관 사칭형",
            summary="국민건강보험을 사칭한 스미싱 문자",
        ),
        indicators=["즉시 확인 유도"],
        messages=messages
        or [ChatMessage(role=ChatRole.USER, content="이거 진짜인가요?")],
    )


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", True)
async def test_get_response_mock_mode_bypasses_gemini():
    service = ChatService()

    response = await service.get_response(_request())

    assert "금융감독원" in response.message


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
@patch("app.chat.service.GEMINI_API_KEY", None)
async def test_get_response_raises_when_api_key_missing():
    service = ChatService()

    with pytest.raises(ChatServiceError, match="Missing API Key"):
        await service.get_response(_request())


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
@patch("app.chat.service.GEMINI_API_KEY", "fake-key")
async def test_get_response_returns_gemini_text_and_maps_assistant_role():
    history = [
        ChatMessage(role=ChatRole.USER, content="이거 진짜인가요?"),
        ChatMessage(role=ChatRole.ASSISTANT, content="네, 의심스러운 문자입니다."),
        ChatMessage(role=ChatRole.USER, content="그럼 어떻게 해야 하나요?"),
    ]

    fake_response = {
        "candidates": [
            {"content": {"parts": [{"text": "금융감독원(1332)에 신고해 주세요."}]}}
        ]
    }

    with patch(
        "app.chat.service.GeminiClient.generate",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as mock_generate:
        service = ChatService()
        response = await service.get_response(_request(messages=history))

    assert response.message == "금융감독원(1332)에 신고해 주세요."

    sent_payload = mock_generate.call_args.kwargs["payload"]
    sent_roles = [content["role"] for content in sent_payload["contents"]]
    assert sent_roles == ["user", "model", "user"]


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
@patch("app.chat.service.GEMINI_API_KEY", "fake-key")
async def test_get_response_raises_on_empty_candidates():
    with patch(
        "app.chat.service.GeminiClient.generate",
        new_callable=AsyncMock,
        return_value={"candidates": []},
    ):
        service = ChatService()
        with pytest.raises(ChatServiceError, match="Empty Response"):
            await service.get_response(_request())


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
@patch("app.chat.service.GEMINI_API_KEY", "fake-key")
async def test_get_response_raises_on_rate_limit():
    error = httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("POST", "https://example.com"),
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
    )

    with patch(
        "app.chat.service.GeminiClient.generate",
        new_callable=AsyncMock,
        side_effect=error,
    ):
        service = ChatService()
        with pytest.raises(ChatServiceError, match="Rate Limit"):
            await service.get_response(_request())


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
@patch("app.chat.service.GEMINI_API_KEY", "fake-key")
async def test_get_response_raises_on_timeout():
    with patch(
        "app.chat.service.GeminiClient.generate",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("timed out"),
    ):
        service = ChatService()
        with pytest.raises(ChatServiceError, match="Timeout"):
            await service.get_response(_request())
