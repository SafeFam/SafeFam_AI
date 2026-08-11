"""Stacking 자체 모델과 Gemini를 결합하는 텍스트 분석기"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.analysis.hybrid_policy import (
    ConditionalGeminiPolicy,
    HybridRoutingDecision,
    HybridRoutingResult,
)
from app.analysis.risk_policy import (
    determine_text_risk_grade,
)

logger = logging.getLogger(__name__)

StackingAnalyzer = Callable[
    [str],
    dict[str, Any],
]

GeminiAnalyzer = Callable[
    [str],
    Awaitable[dict[str, Any]],
]

def _stacking_to_public_result(
    stacking_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Stacking 결과를 기존 텍스트 result 형식으로 변환"""
     
    result = stacking_analysis.get("result") or {}
    risk_score = result.get("risk_score")

    if not isinstance(risk_score, int):
        return {
            "grade": "UNKNOWN",
            "risk_score": None,
            "tone_analysis": "",
            "evidence": [],
            "reason": (
                "Stacking 자체 모델 결과를 "
                "사용할 수 없습니다."
            ),
            "error_message": (
                "STACKING_MODEL_UNAVAILABLE"
            ),
        }

    return {
        "grade": determine_text_risk_grade(
            risk_score
        ),
        "risk_score": risk_score,
        "tone_analysis": "",
        "evidence": [],
        "reason": (
            "Stacking 자체 모델의 확신 구간 "
            "판정을 사용했습니다."
        ),
        "error_message": None,
    }

def _gemini_is_available(
    gemini_analysis: dict[str, Any],
) -> bool:
    """Gemini 응답을 최종 판정에 사용할 수 있는지 검사"""

    result = gemini_analysis.get("result") or {}

    return (
        result.get("grade") != "UNKNOWN"
        and result.get("risk_score") is not None
        and not result.get("error_message")
    )

class HybridTextAnalyzer:
    """Stacking 분석 후 필요한 경우에만 Gemini 호출"""

    def __init__(
        self,
        *,
        policy: ConditionalGeminiPolicy,
        stacking_analyzer: StackingAnalyzer,
        gemini_analyzer: GeminiAnalyzer,
    ) -> None:
        self.policy = policy
        self.stacking_analyzer = stacking_analyzer
        self.gemini_analyzer = gemini_analyzer

    async def analyze(
        self,
        text: str,
        *,
        force_gemini: bool = False,
    ) -> dict[str, Any]:
        """단일 메시지를 조건부 하이브리드 방식으로 분석"""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # 항상 자체 모델을 먼저 실행한다. 분석기 구현이 예외를 그대로
        # 전파하더라도 전체 파이프라인이 중단되지 않도록 unavailable
        # 결과로 정규화한 뒤 Gemini fallback 정책을 적용한다.
        try:
            stacking_analysis = await asyncio.to_thread(
                self.stacking_analyzer,
                text,
            )
        except Exception as exception:
            logger.error(
                "[Hybrid Text] Stacking analyzer failed. "
                "error_type=%s",
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

        # 자체 모델 결과로 Gemini 호출 여부 결정
        routing = self.policy.route(
            stacking_analysis
        )

        # 규칙 엔진이 위험 신호를 발견했거나 실패했다면,
        # Stacking 확신 구간이어도 Gemini 재검증을 수행한다.
        if force_gemini:
            routing = HybridRoutingResult(
                decision=(
                    HybridRoutingDecision.GEMINI_REVIEW
                ),
                should_call_gemini=True,
                reason="RULE_RISK_ESCALATION",
            )

        stacking_result = (
            stacking_analysis.get("result") or {}
        )

        # 자체 모델이 확실하면 Gemini 호출 X
        if not routing.should_call_gemini:
            logger.info(
                "[Hybrid Text] Gemini skipped. "
                "decision=%s risk_score=%s confidence=%s",
                routing.decision.value,
                stacking_result.get("risk_score"),
                stacking_result.get("confidence"),
            )

            return {
                "engine": "hybrid_stacking_gemini",
                "result": _stacking_to_public_result(
                    stacking_analysis
                ),
                "self_model": stacking_result,
                "gemini": None,
                "gemini_called": False,
                "gemini_available": False,
                "decision_source": "STACKING",
                "routing_decision": (
                    routing.decision.value
                ),
                "routing_reason": routing.reason,
                "fallback_applied": False,
            }

        # 불확실 구간 또는 자체 모델 장애일 때 Gemini 호출
        logger.info(
            "[Hybrid Text] Gemini review requested. "
            "decision=%s",
            routing.decision.value,
        )

        # Gemini 분석기가 예상 밖의 예외를 전파해도 사용 가능한
        # stacking 결과로 fallback할 수 있도록 실패 응답으로 정규화한다.
        try:
            gemini_analysis = await self.gemini_analyzer(
                text
            )
        except Exception as exception:
            logger.error(
                "[Hybrid Text] Gemini analyzer failed. "
                "error_type=%s",
                type(exception).__name__,
            )
            gemini_analysis = {
                "result": {
                    "grade": "UNKNOWN",
                    "risk_score": None,
                    "tone_analysis": "",
                    "evidence": [],
                    "reason": "Gemini 분석을 사용할 수 없습니다.",
                    "error_message": "GEMINI_ANALYZER_FAILED",
                }
            }
        gemini_available = _gemini_is_available(
            gemini_analysis
        )

        # Gemini가 성공했다면 Gemini 문맥 판정 사용
        if gemini_available:
            return {
                "engine": "hybrid_stacking_gemini",
                "result": gemini_analysis["result"],
                "self_model": stacking_result,
                "gemini": gemini_analysis["result"],
                "gemini_called": True,
                "gemini_available": True,
                "decision_source": "GEMINI",
                "routing_decision": (
                    routing.decision.value
                ),
                "routing_reason": routing.reason,
                "fallback_applied": False,
            }

        # Gemini는 실패했지만 stacking 결과가 있다면
        # stacking 결과를 보수적으로 유지
        if stacking_analysis.get(
            "is_available",
            False,
        ):
            logger.warning(
                "[Hybrid Text] Gemini failed; "
                "using stacking fallback. decision=%s",
                routing.decision.value,
            )

            return {
                "engine": "hybrid_stacking_gemini",
                "result": _stacking_to_public_result(
                    stacking_analysis
                ),
                "self_model": stacking_result,
                "gemini": (
                    gemini_analysis.get("result")
                ),
                "gemini_called": True,
                "gemini_available": False,
                "decision_source": (
                    "STACKING_FALLBACK"
                ),
                "routing_decision": (
                    routing.decision.value
                ),
                "routing_reason": routing.reason,
                "fallback_applied": True,
            }

        # 두 엔진이 모두 실패했다면 정상 점수 반환 X
        logger.error(
            "[Hybrid Text] All text engines unavailable."
        )

        return {
            "engine": "hybrid_stacking_gemini",
            "result": {
                "grade": "UNKNOWN",
                "risk_score": None,
                "tone_analysis": "",
                "evidence": [],
                "reason": (
                    "모든 텍스트 분석 엔진을 "
                    "사용할 수 없습니다."
                ),
                "error_message": (
                    "ALL_TEXT_ENGINES_UNAVAILABLE"
                ),
            },
            "self_model": stacking_result,
            "gemini": (
                gemini_analysis.get("result")
            ),
            "gemini_called": True,
            "gemini_available": False,
            "decision_source": "UNAVAILABLE",
            "routing_decision": (
                HybridRoutingDecision
                .GEMINI_FALLBACK
                .value
            ),
            "routing_reason": routing.reason,
            "fallback_applied": True,
        }
