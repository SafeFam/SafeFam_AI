import pytest

from app.analysis.hybrid_policy import ConditionalLlmPolicy, HybridThresholds
from app.analysis.text.hybrid_analyzer import (
    HybridTextAnalyzer,
    _stacking_to_public_result,
)


def build_stacking_result(probability: float) -> dict:
    return {
        "engine": "stacking",
        "is_available": True,
        "result": {
            "risk_probability": probability,
            "risk_score": round(probability * 100),
            "confidence": 0.8,
            "error_message": None,
        },
    }


def build_llm_result(score: int = 80) -> dict:
    return {
        "is_mock": False,
        "provider": "AWS_BEDROCK",
        "model_id": "test-model",
        "result": {
            "grade": "DANGEROUS",
            "risk_score": score,
            "tone_analysis": "urgent tone",
            "evidence": ["immediate action"],
            "reason": "smishing is likely",
            "error_message": None,
        },
    }


def build_policy() -> ConditionalLlmPolicy:
    return ConditionalLlmPolicy(HybridThresholds(normal_max=0.2, phishing_min=0.8))


def test_unavailable_stacking_result_uses_localized_reason():
    result = _stacking_to_public_result(
        {"is_available": False, "result": {"risk_score": None}}
    )

    assert result["reason"] == "문자 분석 결과를 확인할 수 없습니다."
    assert result["error_message"] == "STACKING_MODEL_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [0.1, 0.9])
async def test_skips_llm_for_confident_stacking(probability: float):
    calls = 0

    async def llm_analyzer(_text: str):
        nonlocal calls
        calls += 1
        return build_llm_result()

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: build_stacking_result(probability),
        llm_analyzer=llm_analyzer,
    )
    result = await analyzer.analyze("message")

    assert calls == 0
    assert result["llm_called"] is False
    assert result["llm_available"] is False
    assert result["decision_source"] == "STACKING"
    assert result["result"]["reason"] == "문자에 나타난 특징을 분석해 판단했습니다."
    assert result["result"]["error_message"] is None
    assert result["gemini_called"] is False
    assert result["gemini"] is None


@pytest.mark.asyncio
async def test_calls_llm_for_uncertain_prediction():
    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: build_stacking_result(0.5),
        llm_analyzer=lambda _text: _async_result(build_llm_result(85)),
    )
    result = await analyzer.analyze("message")

    assert result["llm_called"] is True
    assert result["llm_available"] is True
    assert result["llm_provider"] == "AWS_BEDROCK"
    assert result["llm_model"] == "test-model"
    assert result["decision_source"] == "LLM"
    assert result["result"]["risk_score"] == 85
    assert result["gemini_called"] is True
    assert result["gemini_available"] is True
    assert result["gemini"] == result["llm"]


@pytest.mark.asyncio
async def test_force_llm_overrides_confident_stacking():
    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: build_stacking_result(0.1),
        llm_analyzer=lambda _text: _async_result(build_llm_result(85)),
    )
    result = await analyzer.analyze("message", force_llm=True)

    assert result["llm_called"] is True
    assert result["routing_decision"] == "LLM_REVIEW"
    assert result["routing_reason"] == "RULE_RISK_ESCALATION"


@pytest.mark.asyncio
async def test_uses_stacking_when_llm_fails():
    failed_llm = {
        "provider": "AWS_BEDROCK",
        "model_id": "test-model",
        "result": {
            "grade": "UNKNOWN",
            "risk_score": None,
            "error_message": "LLM_TIMEOUT",
        },
    }
    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: build_stacking_result(0.5),
        llm_analyzer=lambda _text: _async_result(failed_llm),
    )
    result = await analyzer.analyze("message")

    assert result["decision_source"] == "STACKING_FALLBACK"
    assert result["fallback_applied"] is True
    assert result["result"]["risk_score"] == 50
    assert result["result"]["reason"] == "문자에 나타난 특징을 분석해 판단했습니다."
    assert result["llm_available"] is False


@pytest.mark.asyncio
async def test_uses_llm_when_stacking_raises():
    def raising_stacking(_text: str):
        raise RuntimeError("local model failure")

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=raising_stacking,
        llm_analyzer=lambda _text: _async_result(build_llm_result(75)),
    )
    result = await analyzer.analyze("message")

    assert result["decision_source"] == "LLM"
    assert result["routing_decision"] == "LLM_FALLBACK"
    assert result["result"]["risk_score"] == 75


@pytest.mark.asyncio
async def test_does_not_fail_open_when_all_engines_fail():
    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: {
            "is_available": False,
            "result": {"risk_probability": None, "risk_score": None},
        },
        llm_analyzer=lambda _text: _async_result(
            {
                "result": {
                    "grade": "UNKNOWN",
                    "risk_score": None,
                    "error_message": "LLM_TIMEOUT",
                }
            }
        ),
    )
    result = await analyzer.analyze("message")

    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["risk_score"] is None
    assert result["result"]["reason"] == "현재 문자 내용을 분석할 수 없습니다."
    assert result["result"]["error_message"] == "ALL_TEXT_ENGINES_UNAVAILABLE"


async def _async_result(result: dict) -> dict:
    return result
