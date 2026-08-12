from unittest.mock import patch

import pytest

from app.chat.schemas import ChatMessage, ChatRequest, ChatRole
from app.chat.service import ChatService, ChatServiceError
from app.infrastructure.llm.bedrock_client import LlmProviderError
from app.infrastructure.llm.types import LlmGeneration


def build_request() -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role=ChatRole.USER, content="Is this message safe?")]
    )


class StubClient:
    def __init__(self, text: str = "Use an official contact channel.") -> None:
        self.text = text
        self.kwargs = None

    async def generate(self, **kwargs) -> LlmGeneration:
        self.kwargs = kwargs
        return LlmGeneration(
            text=self.text,
            provider="AWS_BEDROCK",
            model_id="test-model",
        )


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", True)
async def test_chat_mock_mode():
    response = await ChatService().get_response(build_request())
    assert response.message


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
async def test_chat_uses_provider_neutral_client():
    client = StubClient()
    response = await ChatService(client=client).get_response(build_request())

    assert response.message == "Use an official contact channel."
    assert client.kwargs["messages"] == [
        {"role": "user", "content": "Is this message safe?"}
    ]


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
async def test_chat_rejects_empty_response():
    with pytest.raises(ChatServiceError, match="LLM_EMPTY_RESPONSE"):
        await ChatService(client=StubClient("   ")).get_response(build_request())


@pytest.mark.asyncio
@patch("app.chat.service.MOCK_ENABLED", False)
async def test_chat_normalizes_provider_error():
    class FailingClient(StubClient):
        async def generate(self, **_kwargs) -> LlmGeneration:
            raise LlmProviderError("LLM_THROTTLED")

    with pytest.raises(ChatServiceError, match="LLM_THROTTLED"):
        await ChatService(client=FailingClient()).get_response(build_request())
