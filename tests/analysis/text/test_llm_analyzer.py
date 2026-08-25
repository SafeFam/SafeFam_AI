from unittest.mock import patch

import pytest

from app.analysis.text.llm_analyzer import analyze_text_with_llm
from app.infrastructure.llm.bedrock_client import LlmProviderError
from app.infrastructure.llm.types import LlmGeneration


class StubLlmClient:
    provider = "AWS_BEDROCK"
    model_id = "test-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def generate(self, **_kwargs) -> LlmGeneration:
        self.calls += 1
        return LlmGeneration(
            text=self.response,
            provider=self.provider,
            model_id=self.model_id,
        )


@pytest.mark.asyncio
@patch("app.analysis.text.llm_analyzer.MOCK_ENABLED", True)
async def test_analyze_text_with_llm_mock_mode():
    result = await analyze_text_with_llm("test message")

    assert result["is_mock"] is True
    assert result["result"]["grade"] == "DANGEROUS"
    assert result["result"]["risk_score"] == 85


@pytest.mark.asyncio
async def test_analyze_text_with_llm_parses_valid_response():
    client = StubLlmClient(
        '{"risk_score": 82, "tone_analysis": "urgent tone", '
        '"evidence": ["immediate action requested"], '
        '"reason": "institutional impersonation"}'
    )

    result = await analyze_text_with_llm("message", client=client)

    assert client.calls == 1
    assert result["result"]["grade"] == "DANGEROUS"
    assert result["result"]["risk_score"] == 82
    assert result["provider"] == "AWS_BEDROCK"
    assert result["model_id"] == "test-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"risk_score": 101, "tone_analysis": "tone", '
        '"evidence": [], "reason": "reason"}',
        '{"risk_score": 50, "tone_analysis": "tone", '
        '"evidence": "invalid", "reason": "reason"}',
    ],
)
async def test_analyze_text_with_llm_rejects_invalid_response(response: str):
    result = await analyze_text_with_llm(
        "message",
        client=StubLlmClient(response),
    )

    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["risk_score"] is None
    assert result["result"]["error_message"] == "LLM_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_analyze_text_with_llm_normalizes_provider_error():
    class FailingClient(StubLlmClient):
        async def generate(self, **_kwargs) -> LlmGeneration:
            raise LlmProviderError("LLM_THROTTLED")

    result = await analyze_text_with_llm(
        "message",
        client=FailingClient(""),
    )

    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["reason"] == (
        "문맥 분석을 마치지 못해 나머지 검사 결과로만 판단했습니다."
    )
    assert result["result"]["error_message"] == "LLM_THROTTLED"
