"""세 가지 하이브리드 평가 모드 실행기 테스트."""
from __future__ import annotations

from typing import Any

import pytest

from app.analysis.hybrid_policy import ConditionalLlmPolicy, HybridThresholds
from data_science.SMSModel.hybrid_evaluation import (
    EvaluationMode,
    EvaluationSample,
    HybridEvaluationRunner,
)


def _stacking_result(
    probability: float,
    *,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "engine": "stacking",
        "is_available": available,
        "result": {
            "risk_probability": probability if available else None,
            "risk_score": round(probability * 100) if available else None,
            "confidence": 0.9 if available else 0.0,
            "is_suspected_phishing": probability >= 0.5 if available else None,
            "error_message": None if available else "STACKING_MODEL_UNAVAILABLE",
        },
    }


def _llm_result(
    grade: str = "DANGEROUS",
    *,
    error_code: str | None = None,
) -> dict[str, Any]:
    available = grade != "UNKNOWN" and error_code is None
    return {
        "provider": "AWS_BEDROCK",
        "model_id": "anthropic.claude-haiku-4-5",
        "usage": {
            "input_tokens": 120 if available else None,
            "output_tokens": 30 if available else None,
        },
        "result": {
            "grade": grade,
            "risk_score": 85 if available else None,
            "tone_analysis": "test",
            "evidence": [],
            "reason": "test",
            "error_message": error_code,
        },
    }


def _sample() -> EvaluationSample:
    return EvaluationSample(
        sample_id="sha256:test-sample",
        text="평가 중에만 존재하는 메시지",
        expected_label="phishing",
    )


def _policy() -> ConditionalLlmPolicy:
    return ConditionalLlmPolicy(
        HybridThresholds(normal_max=0.2, phishing_min=0.8)
    )


@pytest.mark.asyncio
async def test_self_model_only_never_calls_llm() -> None:
    llm_calls = 0

    async def llm_analyzer(_text: str) -> dict[str, Any]:
        nonlocal llm_calls
        llm_calls += 1
        raise AssertionError("SELF_MODEL_ONLY must not call the LLM")

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.9),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.SELF_MODEL_ONLY)

    assert llm_calls == 0
    assert record.predicted_label == "phishing"
    assert record.result_available is True
    assert record.llm_called is False
    assert record.decision_source == "STACKING"


@pytest.mark.asyncio
async def test_llm_only_never_calls_stacking_and_collects_usage() -> None:
    stacking_calls = 0

    def stacking_analyzer(_text: str) -> dict[str, Any]:
        nonlocal stacking_calls
        stacking_calls += 1
        raise AssertionError("LLM_ONLY must not call Stacking")

    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result("SAFE")

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=stacking_analyzer,
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.LLM_ONLY)

    assert stacking_calls == 0
    assert record.predicted_label == "normal"
    assert record.llm_called is True
    assert record.llm_available is True
    assert record.llm_provider == "AWS_BEDROCK"
    assert record.input_tokens == 120
    assert record.output_tokens == 30


@pytest.mark.asyncio
async def test_hybrid_skips_llm_for_confident_stacking() -> None:
    llm_calls = 0

    async def llm_analyzer(_text: str) -> dict[str, Any]:
        nonlocal llm_calls
        llm_calls += 1
        return _llm_result()

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.1),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.HYBRID)

    assert llm_calls == 0
    assert record.predicted_label == "normal"
    assert record.llm_called is False
    assert record.routing_decision == "SELF_MODEL_NORMAL"
    assert record.routing_reason == "HIGH_CONFIDENCE_NORMAL"


@pytest.mark.asyncio
async def test_hybrid_calls_llm_only_for_uncertain_stacking() -> None:
    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result()

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.5),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.HYBRID)

    assert record.predicted_label == "phishing"
    assert record.llm_called is True
    assert record.llm_available is True
    assert record.decision_source == "LLM"
    assert record.routing_decision == "LLM_REVIEW"
    assert record.input_tokens == 120
    assert record.output_tokens == 30


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_stacking_when_llm_fails() -> None:
    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result("UNKNOWN", error_code="LLM_TIMEOUT")

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.5),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.HYBRID)

    assert record.predicted_label == "phishing"
    assert record.result_available is True
    assert record.llm_available is False
    assert record.fallback_applied is True
    assert record.all_engines_unavailable is False
    assert record.decision_source == "STACKING_FALLBACK"
    assert record.error_code == "LLM_TIMEOUT"


@pytest.mark.asyncio
async def test_hybrid_returns_unknown_when_both_engines_fail() -> None:
    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result("UNKNOWN", error_code="LLM_TIMEOUT")

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.5, available=False),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.HYBRID)

    assert record.predicted_label == "unknown"
    assert record.result_available is False
    assert record.all_engines_unavailable is True
    assert record.decision_source == "UNAVAILABLE"
    assert record.error_code == "ALL_TEXT_ENGINES_UNAVAILABLE"


@pytest.mark.asyncio
async def test_evaluate_preserves_sample_and_mode_order_without_storing_text() -> None:
    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result()

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.9),
        llm_analyzer=llm_analyzer,
    )
    samples = [
        _sample(),
        EvaluationSample("sha256:second", "두 번째 원문", "normal"),
    ]

    records = await runner.evaluate(
        samples,
        modes=(EvaluationMode.SELF_MODEL_ONLY, EvaluationMode.LLM_ONLY),
    )

    assert [(record.sample_id, record.mode) for record in records] == [
        ("sha256:test-sample", EvaluationMode.SELF_MODEL_ONLY),
        ("sha256:test-sample", EvaluationMode.LLM_ONLY),
        ("sha256:second", EvaluationMode.SELF_MODEL_ONLY),
        ("sha256:second", EvaluationMode.LLM_ONLY),
    ]
    assert all("text" not in record.to_dict() for record in records)


@pytest.mark.asyncio
async def test_llm_unknown_is_not_treated_as_normal() -> None:
    async def llm_analyzer(_text: str) -> dict[str, Any]:
        return _llm_result("UNKNOWN", error_code="LLM_INVALID_RESPONSE")

    runner = HybridEvaluationRunner(
        policy=_policy(),
        stacking_analyzer=lambda _text: _stacking_result(0.9),
        llm_analyzer=llm_analyzer,
    )

    record = await runner.evaluate_one(_sample(), EvaluationMode.LLM_ONLY)

    assert record.predicted_label == "unknown"
    assert record.result_available is False
    assert record.all_engines_unavailable is True
    assert record.error_code == "LLM_INVALID_RESPONSE"


@pytest.mark.parametrize(
    "sample",
    [
        ("", "message", "normal"),
        ("id", "message", "unknown"),
    ],
)
def test_rejects_invalid_samples(sample: tuple[str, str, str]) -> None:
    with pytest.raises(ValueError):
        EvaluationSample(*sample)
