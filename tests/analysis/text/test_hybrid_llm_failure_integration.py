"""LLM 공급자 실패가 하이브리드 판정까지 전달되는 통합 회귀 테스트."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from app.analysis.hybrid_policy import ConditionalLlmPolicy, HybridThresholds
from app.analysis.text.hybrid_analyzer import HybridTextAnalyzer
from app.analysis.text.llm_analyzer import analyze_text_with_llm
from app.infrastructure.llm.types import LlmGeneration, LlmProviderError


def _stacking_result(*, available: bool = True) -> dict[str, Any]:
    return {
        "engine": "stacking",
        "is_available": available,
        "result": {
            "risk_probability": 0.5 if available else None,
            "risk_score": 50 if available else None,
            "confidence": 0.5 if available else 0.0,
            "error_message": None if available else "Stacking Inference Error",
        },
    }


class StubLlmClient:
    provider = "AWS_BEDROCK"
    model_id = "anthropic.claude-haiku-test"

    def __init__(self, *, error_code: str | None = None, text: str = "{}") -> None:
        self.error_code = error_code
        self.text = text

    async def generate(self, **_kwargs: Any) -> LlmGeneration:
        if self.error_code is not None:
            raise LlmProviderError(self.error_code)
        return LlmGeneration(
            text=self.text,
            provider=self.provider,
            model_id=self.model_id,
            input_tokens=10,
            output_tokens=5,
            latency_ms=20,
        )


def _analyzer(client: StubLlmClient, *, stacking_available: bool = True):
    async def llm_analyzer(text: str) -> dict[str, Any]:
        return await analyze_text_with_llm(text, client=client)

    return HybridTextAnalyzer(
        policy=ConditionalLlmPolicy(
            HybridThresholds(normal_max=0.2, phishing_min=0.8)
        ),
        stacking_analyzer=lambda _text: _stacking_result(
            available=stacking_available
        ),
        llm_analyzer=llm_analyzer,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client", "expected_error"),
    [
        (StubLlmClient(error_code="LLM_TIMEOUT"), "LLM_TIMEOUT"),
        (StubLlmClient(error_code="LLM_THROTTLED"), "LLM_THROTTLED"),
        (StubLlmClient(text="not-json"), "LLM_INVALID_RESPONSE"),
    ],
)
async def test_llm_failures_fall_back_to_stacking(
    monkeypatch: pytest.MonkeyPatch,
    client: StubLlmClient,
    expected_error: str,
) -> None:
    monkeypatch.setattr("app.analysis.text.llm_analyzer.MOCK_ENABLED", False)

    result = await _analyzer(client).analyze("분석 대상")

    assert result["llm_called"] is True
    assert result["llm_available"] is False
    assert result["llm_provider"] == "AWS_BEDROCK"
    assert result["llm_model"] == "anthropic.claude-haiku-test"
    assert result["llm"]["error_message"] == expected_error
    assert result["fallback_applied"] is True
    assert result["decision_source"] == "STACKING_FALLBACK"
    assert result["result"]["risk_score"] == 50


@pytest.mark.asyncio
async def test_both_engines_unavailable_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.analysis.text.llm_analyzer.MOCK_ENABLED", False)

    result = await _analyzer(
        StubLlmClient(error_code="LLM_TIMEOUT"),
        stacking_available=False,
    ).analyze("분석 대상")

    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["risk_score"] is None
    assert result["result"]["error_message"] == "ALL_TEXT_ENGINES_UNAVAILABLE"


@pytest.mark.asyncio
async def test_failure_logs_do_not_expose_message_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("app.analysis.text.llm_analyzer.MOCK_ENABLED", False)
    secret_token = "review-secret-token-8c31f5"
    sensitive_message = (
        "주민번호 900101-1234567 "
        f"AWS_SECRET_ACCESS_KEY={secret_token}"
    )

    with caplog.at_level(logging.DEBUG):
        await _analyzer(StubLlmClient(text="not-json")).analyze(
            sensitive_message
        )

    assert sensitive_message not in caplog.text
    assert "900101-1234567" not in caplog.text
    assert "AWS_SECRET_ACCESS_KEY" not in caplog.text
    assert secret_token not in caplog.text
