"""Routing policy for the stacking and provider-neutral LLM analyzers."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HybridRoutingDecision(str, Enum):
    SELF_MODEL_NORMAL = "SELF_MODEL_NORMAL"
    LLM_REVIEW = "LLM_REVIEW"
    SELF_MODEL_PHISHING = "SELF_MODEL_PHISHING"
    LLM_FALLBACK = "LLM_FALLBACK"


@dataclass(frozen=True)
class HybridThresholds:
    normal_max: float
    phishing_min: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.normal_max <= 1.0:
            raise ValueError("normal_max must be between 0 and 1")
        if not 0.0 <= self.phishing_min <= 1.0:
            raise ValueError("phishing_min must be between 0 and 1")
        if self.normal_max >= self.phishing_min:
            raise ValueError("normal_max must be smaller than phishing_min")


@dataclass(frozen=True)
class HybridRoutingResult:
    decision: HybridRoutingDecision
    should_call_llm: bool
    reason: str


class ConditionalLlmPolicy:
    """Route uncertain or unavailable stacking results to the configured LLM."""

    def __init__(self, thresholds: HybridThresholds) -> None:
        self.thresholds = thresholds

    def route(self, stacking_analysis: dict[str, Any]) -> HybridRoutingResult:
        if not stacking_analysis.get("is_available", False):
            return HybridRoutingResult(
                decision=HybridRoutingDecision.LLM_FALLBACK,
                should_call_llm=True,
                reason="STACKING_MODEL_UNAVAILABLE",
            )

        result = stacking_analysis.get("result") or {}
        probability = result.get("risk_probability")
        risk_score = result.get("risk_score")

        if (
            isinstance(risk_score, bool)
            or not isinstance(risk_score, int)
            or not 0 <= risk_score <= 100
        ):
            return HybridRoutingResult(
                decision=HybridRoutingDecision.LLM_FALLBACK,
                should_call_llm=True,
                reason="INVALID_STACKING_SCORE",
            )

        if isinstance(probability, bool) or not isinstance(
            probability, (int, float)
        ):
            return HybridRoutingResult(
                decision=HybridRoutingDecision.LLM_FALLBACK,
                should_call_llm=True,
                reason="STACKING_PROBABILITY_UNAVAILABLE",
            )

        probability = float(probability)
        if not 0.0 <= probability <= 1.0:
            return HybridRoutingResult(
                decision=HybridRoutingDecision.LLM_FALLBACK,
                should_call_llm=True,
                reason="INVALID_STACKING_PROBABILITY",
            )

        if probability <= self.thresholds.normal_max:
            return HybridRoutingResult(
                decision=HybridRoutingDecision.SELF_MODEL_NORMAL,
                should_call_llm=False,
                reason="HIGH_CONFIDENCE_NORMAL",
            )

        if probability >= self.thresholds.phishing_min:
            return HybridRoutingResult(
                decision=HybridRoutingDecision.SELF_MODEL_PHISHING,
                should_call_llm=False,
                reason="HIGH_CONFIDENCE_PHISHING",
            )

        return HybridRoutingResult(
            decision=HybridRoutingDecision.LLM_REVIEW,
            should_call_llm=True,
            reason="UNCERTAIN_SELF_MODEL_PREDICTION",
        )
