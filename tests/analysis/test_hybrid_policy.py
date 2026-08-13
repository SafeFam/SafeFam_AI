"""Gemini 조건부 호출 정책 테스트"""

import pytest

from app.analysis.hybrid_policy import (
    ConditionalLlmPolicy,
    HybridRoutingDecision,
    HybridThresholds,
)


@pytest.fixture
def policy() -> ConditionalLlmPolicy:
    """테스트에서만 사용하는 임시 경계값"""

    return ConditionalLlmPolicy(
        HybridThresholds(
            normal_max=0.2,
            phishing_min=0.8,
        )
    )


def build_stacking_result(
    probability: float,
) -> dict:
    """사용 가능한 stacking 결과 생성"""

    return {
        "engine": "stacking",
        "is_available": True,
        "result": {
            "risk_probability": probability,
            "risk_score": round(probability * 100),
            "confidence": 0.8,
        },
    }


def test_skips_llm_for_confident_normal(
    policy: ConditionalLlmPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.1)
    )

    assert (
        result.decision
        == HybridRoutingDecision.SELF_MODEL_NORMAL
    )
    assert result.should_call_llm is False


def test_calls_llm_for_uncertain_prediction(
    policy: ConditionalLlmPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.5)
    )

    assert (
        result.decision
        == HybridRoutingDecision.LLM_REVIEW
    )
    assert result.should_call_llm is True


def test_skips_llm_for_confident_phishing(
    policy: ConditionalLlmPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.9)
    )

    assert (
        result.decision
        == HybridRoutingDecision.SELF_MODEL_PHISHING
    )
    assert result.should_call_llm is False


@pytest.mark.parametrize(
    ("probability", "expected_decision", "should_call_llm"),
    [
        (0.2 - 1e-9, HybridRoutingDecision.SELF_MODEL_NORMAL, False),
        (0.2, HybridRoutingDecision.SELF_MODEL_NORMAL, False),
        (0.2 + 1e-9, HybridRoutingDecision.LLM_REVIEW, True),
        (0.8 - 1e-9, HybridRoutingDecision.LLM_REVIEW, True),
        (0.8, HybridRoutingDecision.SELF_MODEL_PHISHING, False),
        (0.8 + 1e-9, HybridRoutingDecision.SELF_MODEL_PHISHING, False),
    ],
)
def test_routes_values_immediately_below_at_and_above_boundaries(
    policy: ConditionalLlmPolicy,
    probability: float,
    expected_decision: HybridRoutingDecision,
    should_call_llm: bool,
) -> None:
    result = policy.route(build_stacking_result(probability))

    assert result.decision == expected_decision
    assert result.should_call_llm is should_call_llm


def test_calls_llm_when_stacking_is_unavailable(
    policy: ConditionalLlmPolicy,
) -> None:
    result = policy.route(
        {
            "engine": "stacking",
            "is_available": False,
            "result": {
                "risk_probability": None,
            },
        }
    )

    assert (
        result.decision
        == HybridRoutingDecision.LLM_FALLBACK
    )
    assert result.should_call_llm is True


@pytest.mark.parametrize("risk_score", [True, -1, 101, 10.5, None])
def test_calls_llm_for_invalid_stacking_score(
    policy: ConditionalLlmPolicy,
    risk_score: object,
) -> None:
    stacking = build_stacking_result(0.1)
    stacking["result"]["risk_score"] = risk_score

    result = policy.route(stacking)

    assert result.decision == HybridRoutingDecision.LLM_FALLBACK
    assert result.should_call_llm is True
    assert result.reason == "INVALID_STACKING_SCORE"


@pytest.mark.parametrize(
    ("normal_max", "phishing_min"),
    [
        (-0.1, 0.8),
        (0.2, 1.1),
        (0.8, 0.8),
        (0.9, 0.8),
    ],
)
def test_rejects_invalid_thresholds(
    normal_max: float,
    phishing_min: float,
) -> None:
    with pytest.raises(ValueError):
        HybridThresholds(
            normal_max=normal_max,
            phishing_min=phishing_min,
        )
