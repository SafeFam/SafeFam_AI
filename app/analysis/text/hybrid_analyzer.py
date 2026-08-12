"""Hybrid text analyzer combining stacking with a provider-neutral LLM."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.analysis.hybrid_policy import (
    ConditionalLlmPolicy,
    HybridRoutingDecision,
    HybridRoutingResult,
)
from app.analysis.risk_policy import determine_text_risk_grade

logger = logging.getLogger(__name__)

StackingAnalyzer = Callable[[str], dict[str, Any]]
LlmAnalyzer = Callable[[str], Awaitable[dict[str, Any]]]


def _stacking_to_public_result(
    stacking_analysis: dict[str, Any],
) -> dict[str, Any]:
    result = stacking_analysis.get("result") or {}
    risk_score = result.get("risk_score")

    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int)
        or not 0 <= risk_score <= 100
    ):
        return {
            "grade": "UNKNOWN",
            "risk_score": None,
            "tone_analysis": "",
            "evidence": [],
            "reason": "The stacking model result is unavailable.",
            "error_message": "STACKING_MODEL_UNAVAILABLE",
        }

    return {
        "grade": determine_text_risk_grade(risk_score),
        "risk_score": risk_score,
        "tone_analysis": "",
        "evidence": [],
        "reason": "The confident stacking model decision was used.",
        "error_message": None,
    }


def _llm_is_available(llm_analysis: dict[str, Any]) -> bool:
    result = llm_analysis.get("result") or {}
    risk_score = result.get("risk_score")
    return (
        result.get("grade") != "UNKNOWN"
        and isinstance(risk_score, int)
        and not isinstance(risk_score, bool)
        and 0 <= risk_score <= 100
        and not result.get("error_message")
    )


def _with_legacy_aliases(result: dict[str, Any]) -> dict[str, Any]:
    """Maintain temporary compatibility with legacy SafeFam_BE events."""
    result["gemini_called"] = result.get("llm_called", False)
    result["gemini_available"] = result.get("llm_available", False)
    result["gemini"] = result.get("llm")
    return result


class HybridTextAnalyzer:
    """Call the LLM only for uncertain or unavailable stacking predictions."""

    def __init__(
        self,
        *,
        policy: ConditionalLlmPolicy,
        stacking_analyzer: StackingAnalyzer,
        llm_analyzer: LlmAnalyzer,
    ) -> None:
        self.policy = policy
        self.stacking_analyzer = stacking_analyzer
        self.llm_analyzer = llm_analyzer

    async def analyze(
        self,
        text: str,
        *,
        force_llm: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        try:
            stacking_analysis = await asyncio.to_thread(
                self.stacking_analyzer,
                text,
            )
        except Exception as exception:
            logger.error(
                "[Hybrid Text] Stacking analyzer failed. error_type=%s",
                type(exception).__name__,
            )
            stacking_analysis = {
                "engine": "stacking",
                "is_available": False,
                "result": {
                    "risk_probability": None,
                    "risk_score": None,
                    "confidence": 0.0,
                    "error_message": "STACKING_MODEL_UNAVAILABLE",
                },
            }

        routing = self.policy.route(stacking_analysis)
        if force_llm:
            routing = HybridRoutingResult(
                decision=HybridRoutingDecision.LLM_REVIEW,
                should_call_llm=True,
                reason="RULE_RISK_ESCALATION",
            )

        stacking_result = stacking_analysis.get("result") or {}
        if not routing.should_call_llm:
            logger.info(
                "[Hybrid Text] LLM skipped. decision=%s risk_score=%s "
                "confidence=%s",
                routing.decision.value,
                stacking_result.get("risk_score"),
                stacking_result.get("confidence"),
            )
            return _with_legacy_aliases(
                {
                    "engine": "hybrid_stacking_llm",
                    "result": _stacking_to_public_result(stacking_analysis),
                    "self_model": stacking_result,
                    "llm": None,
                    "llm_called": False,
                    "llm_available": False,
                    "llm_provider": None,
                    "llm_model": None,
                    "llm_usage": {
                        "input_tokens": None,
                        "output_tokens": None,
                    },
                    "decision_source": "STACKING",
                    "routing_decision": routing.decision.value,
                    "routing_reason": routing.reason,
                    "fallback_applied": False,
                }
            )

        logger.info(
            "[Hybrid Text] LLM review requested. decision=%s",
            routing.decision.value,
        )
        try:
            llm_analysis = await self.llm_analyzer(text)
        except Exception as exception:
            logger.error(
                "[Hybrid Text] LLM analyzer failed. error_type=%s",
                type(exception).__name__,
            )
            llm_analysis = {
                "provider": None,
                "model_id": None,
                "result": {
                    "grade": "UNKNOWN",
                    "risk_score": None,
                    "tone_analysis": "",
                    "evidence": [],
                    "reason": "The LLM analysis is unavailable.",
                    "error_message": "LLM_ANALYZER_FAILED",
                },
            }

        llm_result = llm_analysis.get("result") or {}
        llm_available = _llm_is_available(llm_analysis)
        common = {
            "engine": "hybrid_stacking_llm",
            "self_model": stacking_result,
            "llm": llm_result,
            "llm_called": True,
            "llm_available": llm_available,
            "llm_provider": llm_analysis.get("provider"),
            "llm_model": llm_analysis.get("model_id"),
            "llm_usage": llm_analysis.get("usage") or {
                "input_tokens": None,
                "output_tokens": None,
            },
            "routing_decision": routing.decision.value,
            "routing_reason": routing.reason,
        }

        if llm_available:
            return _with_legacy_aliases(
                {
                    **common,
                    "result": llm_result,
                    "decision_source": "LLM",
                    "fallback_applied": False,
                }
            )

        if stacking_analysis.get("is_available", False):
            logger.warning(
                "[Hybrid Text] LLM failed; using stacking fallback. decision=%s",
                routing.decision.value,
            )
            return _with_legacy_aliases(
                {
                    **common,
                    "result": _stacking_to_public_result(stacking_analysis),
                    "decision_source": "STACKING_FALLBACK",
                    "fallback_applied": True,
                }
            )

        logger.error("[Hybrid Text] All text engines unavailable.")
        return _with_legacy_aliases(
            {
                **common,
                "result": {
                    "grade": "UNKNOWN",
                    "risk_score": None,
                    "tone_analysis": "",
                    "evidence": [],
                    "reason": "All text analysis engines are unavailable.",
                    "error_message": "ALL_TEXT_ENGINES_UNAVAILABLE",
                },
                "decision_source": "UNAVAILABLE",
                "routing_decision": HybridRoutingDecision.LLM_FALLBACK.value,
                "fallback_applied": True,
            }
        )
