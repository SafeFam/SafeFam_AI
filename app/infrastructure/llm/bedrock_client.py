"""AWS Bedrock Converse API 클라이언트"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
)

from app.core.config import settings
from app.infrastructure.llm.types import (
    LlmGeneration,
    LlmProviderError,
)

logger = logging.getLogger(__name__)


class BedrockLlmClient:
    """Bedrock Converse API 어댑터"""

    provider = "AWS_BEDROCK"

    def __init__(
        self,
        *,
        client: Any | None = None,
    ) -> None:
        self.model_id = settings.BEDROCK_MODEL_ID
        self._total_timeout_seconds = (
            settings.LLM_TIMEOUT_SECONDS
            * (settings.LLM_MAX_RETRIES + 1)
        )

        if client is not None:
            self._client = client
            return

        session_kwargs: dict[str, str] = {}

        # 로컬 SSO에서만 사용
        # 운영에서는 IAM Role이 자동 선택
        if settings.AWS_PROFILE:
            session_kwargs["profile_name"] = (
                settings.AWS_PROFILE
            )

        session = boto3.Session(
            **session_kwargs
        )

        self._client = session.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            config=Config(
                connect_timeout=(
                    settings.LLM_TIMEOUT_SECONDS
                ),
                read_timeout=(
                    settings.LLM_TIMEOUT_SECONDS
                ),
                retries={
                    # 최초 호출 + 설정된 retry 수
                    "total_max_attempts": (
                        settings.LLM_MAX_RETRIES + 1
                    ),
                    "mode": "standard",
                },
            ),
        )

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LlmGeneration:
        """동기 boto3 호출을 worker thread에서 실행"""

        if not isinstance(system_prompt, str):
            raise TypeError(
                "system_prompt must be a string"
            )

        if not messages:
            raise ValueError(
                "at least one message is required"
            )

        bedrock_messages = self._convert_messages(
            messages
        )

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.converse,
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=bedrock_messages,
                    inferenceConfig={
                        "maxTokens": (
                            max_tokens
                            if max_tokens is not None
                            else settings.LLM_MAX_OUTPUT_TOKENS
                        ),
                        "temperature": (
                            temperature
                            if temperature is not None
                            else settings.LLM_TEMPERATURE
                        ),
                    },
                ),
                timeout=self._total_timeout_seconds,
            )

        except TimeoutError as exception:
            logger.error("[LLM] Bedrock request exceeded the total deadline.")
            raise LlmProviderError("LLM_TIMEOUT") from exception

        except ClientError as exception:
            error = (
                exception.response.get("Error") or {}
            )
            error_code = str(
                error.get("Code") or "ClientError"
            )

            logger.error(
                "[LLM] Bedrock request failed. "
                "error_code=%s",
                error_code,
            )

            raise LlmProviderError(
                self._public_error_code(
                    error_code
                )
            ) from exception

        except BotoCoreError as exception:
            logger.error(
                "[LLM] Bedrock SDK failed. "
                "error_type=%s",
                type(exception).__name__,
            )

            raise LlmProviderError(
                "LLM_PROVIDER_ERROR"
            ) from exception

        return self._parse_response(
            response
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        expected_role = "user"

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in {
                "user",
                "assistant",
            }:
                raise ValueError(
                    f"unsupported LLM role: {role}"
                )

            if role != expected_role:
                raise ValueError(
                    "LLM messages must start with user and alternate roles"
                )

            if not isinstance(content, str):
                raise TypeError(
                    "message content must be a string"
                )

            converted.append(
                {
                    "role": role,
                    "content": [
                        {
                            "text": content,
                        }
                    ],
                }
            )
            expected_role = "assistant" if role == "user" else "user"

        return converted

    def _parse_response(
        self,
        response: dict[str, Any],
    ) -> LlmGeneration:
        try:
            content = response["output"][
                "message"
            ]["content"]

            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and isinstance(
                    block.get("text"),
                    str,
                )
            ]

            if not text_parts:
                raise ValueError(
                    "Bedrock response has no text"
                )

            usage = response.get("usage") or {}
            metrics = (
                response.get("metrics") or {}
            )

            return LlmGeneration(
                text="".join(text_parts),
                provider=self.provider,
                model_id=self.model_id,
                input_tokens=self._optional_int(
                    usage.get("inputTokens")
                ),
                output_tokens=self._optional_int(
                    usage.get("outputTokens")
                ),
                latency_ms=self._optional_int(
                    metrics.get("latencyMs")
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exception:
            logger.error(
                "[LLM] Bedrock response parsing "
                "failed. error_type=%s",
                type(exception).__name__,
            )

            raise LlmProviderError(
                "LLM_INVALID_RESPONSE"
            ) from exception

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> int | None:
        if isinstance(value, bool):
            return None

        if (
            isinstance(value, int)
            and value >= 0
        ):
            return value

        return None

    @staticmethod
    def _public_error_code(
        error_code: str,
    ) -> str:
        if error_code == "ThrottlingException":
            return "LLM_THROTTLED"

        if error_code in {
            "ModelTimeoutException",
            "ModelNotReadyException",
        }:
            return "LLM_TIMEOUT"

        if error_code in {
            "AccessDeniedException",
            "UnrecognizedClientException",
        }:
            return "LLM_ACCESS_DENIED"

        return "LLM_PROVIDER_ERROR"
