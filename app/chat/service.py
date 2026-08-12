"""Stateless chat service backed by the configured LLM provider."""
from __future__ import annotations

import logging

from app.chat.prompts import build_system_prompt
from app.chat.schemas import ChatMessage, ChatRequest, ChatResponse
from app.core.config import settings
from app.infrastructure.llm.bedrock_client import LlmProviderError
from app.infrastructure.llm.factory import get_llm_client
from app.infrastructure.llm.types import LlmClient

logger = logging.getLogger(__name__)

MOCK_ENABLED = settings.MOCK_SECURITY_API
MOCK_RESPONSE_MESSAGE = (
    "[Mock response] Verify the sender through an official channel. "
    "Never disclose passwords or verification codes."
)


class ChatServiceError(Exception):
    """Raised when the configured LLM cannot generate a chat response."""


def _build_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in messages
    ]


class ChatService:
    """Generate a response from analysis context and conversation history."""

    def __init__(self, client: LlmClient | None = None) -> None:
        self._client = client

    async def get_response(self, request: ChatRequest) -> ChatResponse:
        if MOCK_ENABLED:
            logger.info("[Mock LLM Chat] Provider call skipped.")
            return ChatResponse(message=MOCK_RESPONSE_MESSAGE)

        try:
            client = self._client or get_llm_client()
            generation = await client.generate(
                system_prompt=build_system_prompt(request.analysisContext),
                messages=_build_messages(request.messages),
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
            )
            message_text = generation.text.strip()
            if not message_text:
                logger.error("[LLM Chat] Provider returned an empty response.")
                raise ChatServiceError("LLM_EMPTY_RESPONSE")

            logger.info(
                "[LLM Chat] Response completed. provider=%s model=%s",
                generation.provider,
                generation.model_id,
            )
            return ChatResponse(message=message_text)
        except ChatServiceError:
            raise
        except LlmProviderError as exception:
            error_code = str(exception) or "LLM_PROVIDER_ERROR"
            logger.error("[LLM Chat] Provider failed. error_code=%s", error_code)
            raise ChatServiceError(error_code) from exception
        except Exception as exception:
            logger.error(
                "[LLM Chat] Unexpected failure. error_type=%s",
                type(exception).__name__,
            )
            raise ChatServiceError("LLM_CHAT_FAILED") from exception
