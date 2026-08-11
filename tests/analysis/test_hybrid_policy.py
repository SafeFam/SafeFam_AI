"""Gemini 조건부 호출 정책 테스트"""

import pytest

from app.analysis.hybrid_policy import (
    ConditionalGeminiPolicy,
    HybridRoutingDecision,
    HybridThresholds,
)


@pytest.fixture
def policy() -> ConditionalGeminiPolicy:
    """테스트에서만 사용하는 임시 경계값"""

    return ConditionalGeminiPolicy(
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


def test_skips_gemini_for_confident_normal(
    policy: ConditionalGeminiPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.1)
    )

    assert (
        result.decision
        == HybridRoutingDecision.SELF_MODEL_NORMAL
    )
    assert result.should_call_gemini is False


def test_calls_gemini_for_uncertain_prediction(
    policy: ConditionalGeminiPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.5)
    )

    assert (
        result.decision
        == HybridRoutingDecision.GEMINI_REVIEW
    )
    assert result.should_call_gemini is True


def test_skips_gemini_for_confident_phishing(
    policy: ConditionalGeminiPolicy,
) -> None:
    result = policy.route(
        build_stacking_result(0.9)
    )

    assert (
        result.decision
        == HybridRoutingDecision.SELF_MODEL_PHISHING
    )
    assert result.should_call_gemini is False


def test_calls_gemini_when_stacking_is_unavailable(
    policy: ConditionalGeminiPolicy,
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
        == HybridRoutingDecision.GEMINI_FALLBACK
    )
    assert result.should_call_gemini is True


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