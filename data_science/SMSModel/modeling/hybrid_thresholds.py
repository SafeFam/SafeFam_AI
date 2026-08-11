"""Validation 결과를 이용한 Gemini 조건부 호출 임계값 선정"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)

dataclass(frozen=True)
class HybridThresholdSelection:
    """Validation 데이터에서 선택한 하이브리드 임계값과 성능"""

    normal_probability_max: float
    phishing_probability_min: float

    precision: float
    recall: float
    f2: float

    gemini_call_rate: float
    gemini_call_count: int
    validation_count: int

    target_recall: float
    target_recall_met: bool

    def to_dict(self) -> dict:
        """JSON metadata에 기록할 수 있는 dictionary로 변환"""

        return asdict(self)

def _validate_inputs(
    *,
    stacking_probabilities: np.ndarray,
    gemini_scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
    gemini_phishing_score: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    """임계값 탐색 입력값 검증 및 정규화"""
    probabilities = np.asarray(
        stacking_probabilities,
        dtype=np.float64,
    )

    scores = np.asarray(
        gemini_scores,
        dtype=np.float64,
    )

    normalized_labels = np.asarray(
        labels,
        dtype=str,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "stacking_probabilities must be one-dimensional"
        )

    if scores.ndim != 1:
        raise ValueError(
            "gemini_scores must be one-dimensional"
        )

    if normalized_labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional"
        )

    if len(probabilities) == 0:
        raise ValueError(
            "validation data must not be empty"
        )

    if not (
        len(probabilities)
        == len(scores)
        == len(normalized_labels)
    ):
        raise ValueError(
            "probabilities, Gemini scores and labels "
            "must have the same length"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "stacking probabilities must be finite"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Gemini scores must be finite"
        )

    if (
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ).any():
        raise ValueError(
            "stacking probabilities must be between 0 and 1"
        )

    if (
        (scores < 0.0)
        | (scores > 100.0)
    ).any():
        raise ValueError(
            "Gemini scores must be between 0 and 100"
        )

    observed_labels = set(normalized_labels)

    if observed_labels != {
        "normal",
        "phishing",
    }:
        raise ValueError(
            "validation labels must contain "
            "normal and phishing"
        )

    if not 0.0 < target_recall <= 1.0:
        raise ValueError(
            "target_recall must be between 0 and 1"
        )

    if not 0 <= gemini_phishing_score <= 100:
        raise ValueError(
            "gemini_phishing_score must be "
            "between 0 and 100"
        )

    return (
        probabilities,
        scores,
        normalized_labels,
    )

def select_hybrid_thresholds(
    *,
    stacking_probabilities: np.ndarray,
    gemini_scores: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.95,
    gemini_phishing_score: int = 40,
) -> HybridThresholdSelection:

    """Recall 목표를 만족하면서 F2가 높은 하이브리드 구간 선택"""
    (
        probabilities,
        scores,
        normalized_labels,
    ) = _validate_inputs(
        stacking_probabilities=(
            stacking_probabilities
        ),
        gemini_scores=gemini_scores,
        labels=labels,
        target_recall=target_recall,
        gemini_phishing_score=(
            gemini_phishing_score
        ),
    )

    binary_labels = (
        normalized_labels == "phishing"
    ).astype(np.int64)

    # Gemini의 SAFE/SUSPICIOUS 경계인 40점을 이진 피싱 판정 기준으로 사용
    gemini_predictions = (
        scores >= gemini_phishing_score
    ).astype(np.int64)

    # 고정 간격 후보와 실제 확률값을 함께 사용
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.0,
                    1.0,
                    101,
                ),
                probabilities,
            ]
        )
    )

    best_selection: (
        HybridThresholdSelection | None
    ) = None

    # 비교 우선순위: 
    # F2가 높을수록 좋음
    # Gemini 호출률은 낮을수록 좋음
    # Recall은 높을수록 좋음
    best_rank: (
        tuple[float, float, float] | None
    ) = None

    for normal_max in candidates:
        for phishing_min in candidates:
            if normal_max >= phishing_min:
                continue

            # 두 경계값 사이만 Gemini 호출 대상
            uncertain_mask = (
                (probabilities > normal_max)
                & (
                    probabilities
                    < phishing_min
                )
            )

            # 기본값은 stacking 확률에 따른 자동 판정
            hybrid_predictions = np.where(
                probabilities >= phishing_min,
                1,
                0,
            ).astype(np.int64)

            # 불확실한 구간만 Gemini 결과로 교체
            hybrid_predictions[
                uncertain_mask
            ] = gemini_predictions[
                uncertain_mask
            ]

            recall = float(
                recall_score(
                    binary_labels,
                    hybrid_predictions,
                    zero_division=0,
                )
            )

            if recall < target_recall:
                continue

            precision = float(
                precision_score(
                    binary_labels,
                    hybrid_predictions,
                    zero_division=0,
                )
            )

            f2 = float(
                fbeta_score(
                    binary_labels,
                    hybrid_predictions,
                    beta=2,
                    zero_division=0,
                )
            )

            gemini_call_count = int(
                uncertain_mask.sum()
            )

            gemini_call_rate = float(
                uncertain_mask.mean()
            )

            selection = HybridThresholdSelection(
                normal_probability_max=float(
                    normal_max
                ),
                phishing_probability_min=float(
                    phishing_min
                ),
                precision=precision,
                recall=recall,
                f2=f2,
                gemini_call_rate=(
                    gemini_call_rate
                ),
                gemini_call_count=(
                    gemini_call_count
                ),
                validation_count=len(
                    probabilities
                ),
                target_recall=target_recall,
                target_recall_met=True,
            )

            rank = (
                f2,
                -gemini_call_rate,
                recall,
            )

            if (
                best_rank is None
                or rank > best_rank
            ):
                best_rank = rank
                best_selection = selection

    if best_selection is None:
        raise RuntimeError(
            "no hybrid threshold pair satisfies "
            f"target recall {target_recall:.4f}"
        )

    return best_selection