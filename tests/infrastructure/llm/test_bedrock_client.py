from typing import Any

import pytest
from botocore.exceptions import ClientError

from app.infrastructure.llm.bedrock_client import (
    BedrockLlmClient,
    LlmProviderError,
)


def _response(*text_parts: str) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [{"text": text} for text in text_parts],
            }
        },
        "usage": {"inputTokens": 12, "outputTokens": 7},
        "metrics": {"latencyMs": 123},
    }


class StubBedrockRuntime:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = _response("result") if response is None else response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_generate_builds_converse_request_and_parses_metadata():
    runtime = StubBedrockRuntime(response=_response("hello", " world"))
    client = BedrockLlmClient(client=runtime)

    generation = await client.generate(
        system_prompt="system prompt",
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "previous answer"},
        ],
        max_tokens=256,
        temperature=0.2,
    )

    assert generation.text == "hello world"
    assert generation.provider == "AWS_BEDROCK"
    assert generation.model_id == client.model_id
    assert generation.input_tokens == 12
    assert generation.output_tokens == 7
    assert generation.latency_ms == 123
    assert runtime.calls == [
        {
            "modelId": client.model_id,
            "system": [{"text": "system prompt"}],
            "messages": [
                {"role": "user", "content": [{"text": "question"}]},
                {
                    "role": "assistant",
                    "content": [{"text": "previous answer"}],
                },
            ],
            "inferenceConfig": {"maxTokens": 256, "temperature": 0.2},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "public_code"),
    [
        ("ThrottlingException", "LLM_THROTTLED"),
        ("ModelTimeoutException", "LLM_TIMEOUT"),
        ("ModelNotReadyException", "LLM_TIMEOUT"),
        ("AccessDeniedException", "LLM_ACCESS_DENIED"),
        ("ValidationException", "LLM_PROVIDER_ERROR"),
    ],
)
async def test_generate_normalizes_bedrock_client_errors(
    error_code: str,
    public_code: str,
):
    error = ClientError(
        {"Error": {"Code": error_code, "Message": "sensitive detail"}},
        "Converse",
    )
    client = BedrockLlmClient(client=StubBedrockRuntime(error=error))

    with pytest.raises(LlmProviderError, match=f"^{public_code}$"):
        await client.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "message"}],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {},
        {"output": {"message": {"content": []}}},
        {"output": {"message": {"content": [{"image": {}}]}}},
    ],
)
async def test_generate_rejects_invalid_provider_response(response: dict):
    client = BedrockLlmClient(client=StubBedrockRuntime(response=response))

    with pytest.raises(LlmProviderError, match="^LLM_INVALID_RESPONSE$"):
        await client.generate(
            system_prompt="system",
            messages=[{"role": "user", "content": "message"}],
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_messages_before_provider_call():
    runtime = StubBedrockRuntime()
    client = BedrockLlmClient(client=runtime)

    with pytest.raises(ValueError, match="at least one message"):
        await client.generate(system_prompt="system", messages=[])

    assert runtime.calls == []


@pytest.mark.parametrize("role", ["model", "system", "tool"])
def test_convert_messages_rejects_unsupported_roles(role: str):
    with pytest.raises(ValueError, match="unsupported LLM role"):
        BedrockLlmClient._convert_messages(
            [{"role": role, "content": "message"}]
        )


def test_optional_int_rejects_boolean_negative_and_non_integer_values():
    assert BedrockLlmClient._optional_int(True) is None
    assert BedrockLlmClient._optional_int(-1) is None
    assert BedrockLlmClient._optional_int(1.5) is None
    assert BedrockLlmClient._optional_int(0) == 0
