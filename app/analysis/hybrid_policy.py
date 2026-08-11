"""Stacking 자체 모델의 결과를 Gemini 호출 여부로 변환하는 정책"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

class HybridRoutingDecision(str, Enum):
    """하이브리드 텍스트 분석기의 라우팅 결과"""

    # 자체 모델이 확실한 정상으로 판단
    SELF_MODEL_NORMAL = "SELF_MODEL_NORMAL"

    # 자체 모델 결과가 불확실하여 Gemini 재검증 필요
    GEMINI_REVIEW = "GEMINI_REVIEW"

    # 자체 모델이 확실한 피싱으로 판단
    SELF_MODEL_PHISHING = "SELF_MODEL_PHISHING"

    # 자체 모델을 사용할 수 없어 Gemini로 fallback
    GEMINI_FALLBACK = "GEMINI_FALLBACK"

@dataclass(frozen=True)
class HybridThresholds:
    """Gemini 호출 여부를 결정하는 확률 경계값"""

    # normal_max 이하: 자체 모델의 확실한 정상 판전
    normal_max: float

    # phishing_min 이상: 자체 모델의 확실한 위험 판정
    phishing_min: float

    # normal_max보다 크고 phishing_min보다 작은 구간: 불확실한 구간이므로 Gemini 검증

    def __post_init__(self) -> None:
        """잘못된 임계값으로 서비스가 실행되지 않도록 검증"""
        
        if not 0.0 <= self.normal_max <= 1.0:
            raise ValueError(
                "normal_max must be between 0 and 1"
            )

        if not 0.0 <= self.phishing_min <= 1.0:
            raise ValueError(
                "phishing_min must be between 0 and 1"
            )

        if self.normal_max >= self.phishing_min:
            raise ValueError(
                "normal_max must be smaller than phishing_min"
            )

@dataclass(frozen=True)
class HybridRoutingResult:
    """조건부 호출 정책의 최종 결과"""

    decision: HybridRoutingDecision
    should_call_gemini: bool
    reason: str

class ConditionalGeminiPolicy:
    """Stacking 위험 확률을 Gemini 호출 여부로 변환"""

    def __init__(
        self,
        thresholds: HybridThresholds,
    ) -> None:
        self.thresholds = thresholds

    def route(
        self,
        stacking_analysis: dict[str, Any],
    ) -> HybridRoutingResult:
        """Stacking 분석 결과를 다음 처리 단계로 라우팅"""

        # Stacking artifact 로드 또는 추론에 실패했다면
        # 자체 모델 결과를 신뢰할 수 없으므로 Gemini로 fallback
        if not stacking_analysis.get("is_available", False):
            return HybridRoutingResult(
                decision=(
                    HybridRoutingDecision.GEMINI_FALLBACK
                ),
                should_call_gemini=True,
                reason="STACKING_MODEL_UNAVAILABLE",
            )

        result = stacking_analysis.get("result") or {}
        probability = result.get("risk_probability")

        # 모델은 available이라고 했지만 확률이 없다면
        # 정상으로 간주하지 않고 Gemini로 fallback
        if not isinstance(probability, (int, float)):
            return HybridRoutingResult(
                    decision=(
                        HybridRoutingDecision.GEMINI_FALLBACK
                    ),
                    should_call_gemini=True,
                    reason="STACKING_PROBABILITY_UNAVAILABLE",
                )
        probability = float(probability)

        # NaN 또는 무한대 방지
        if not 0.0 <= probability <= 1.0:
                return HybridRoutingResult(
                    decision=(
                        HybridRoutingDecision.GEMINI_FALLBACK
                    ),
                    should_call_gemini=True,
                    reason="INVALID_STACKING_PROBABILITY",
                )

        if probability <= self.thresholds.normal_max:
            return HybridRoutingResult(
                decision=(
                    HybridRoutingDecision.SELF_MODEL_NORMAL
                ),
                should_call_gemini=False,
                reason="HIGH_CONFIDENCE_NORMAL",
            )

        if probability >= self.thresholds.phishing_min:
            return HybridRoutingResult(
                decision=(
                    HybridRoutingDecision.SELF_MODEL_PHISHING
                ),
                should_call_gemini=False,
                reason="HIGH_CONFIDENCE_PHISHING",
            )

        return HybridRoutingResult(
            decision=HybridRoutingDecision.GEMINI_REVIEW,
            should_call_gemini=True,
            reason="UNCERTAIN_SELF_MODEL_PREDICTION",
        )