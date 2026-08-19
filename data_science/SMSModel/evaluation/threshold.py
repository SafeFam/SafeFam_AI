"""validation 데이터 기반 threshold 선택"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)

PROBABILITY_CANDIDATE_GRID = np.linspace(0.01, 0.99, 99)

@dataclass(frozen=True)
class ThresholdSelection:
    """Validation에서 선택된 threshold와 관련 지표"""

    threshold: float
    precision: float
    recall: float
    f2: float
    false_negative_count: int
    target_recall: float
    target_recall_met: bool


@dataclass(frozen=True)
class ProbabilityThresholdSelection:
    """확률 출력 모델에서 선택된 threshold와 validation 지표"""

    threshold: float
    recall: float
    f2: float
    target_recall: float
    target_recall_met: bool

    def to_validation_metrics(self) -> dict[str, float]:
        """artifact metadata에 기록하는 validation 항목으로 변환"""
        return {
            "recall": self.recall,
            "f2": self.f2,
            "target_recall": self.target_recall,
            "target_recall_met": self.target_recall_met,
        }


def _validate_binary_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> None:
    """label과 score 입력의 기본 무결성을 검사"""
    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    if len(y_true) == 0:
        raise ValueError("cannot select threshold from empty inputs")

    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite numbers")

    allowed_labels = {"normal", "phishing"}

    labels = set(y_true)
    if not labels.issubset(allowed_labels):
        raise ValueError(f"unsupported labels: {labels - allowed_labels}")
    if labels != allowed_labels:
        raise ValueError("y_true must contain both normal and phishing labels")


def _candidate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:
    """validation score에서 결정적인 threshold 후보 생성"""
    unique_scores = np.unique(scores)

    lower_boundary = np.nextafter(
        unique_scores.min(),
        -np.inf,
    )

    return np.concatenate(
        [
            np.asarray([lower_boundary]),
            unique_scores,
        ]
    )


def select_validation_threshold(
    y_true,
    scores,
    *,
    target_recall: float = 0.96,
) -> ThresholdSelection:
    """Validation set에서 Recall 목표를 우선하면서 F2가 가장 높은 threshold를 선택"""
    y_true_array = np.asarray(y_true, dtype=str)
    score_array = np.asarray(scores, dtype=float)

    _validate_binary_inputs(
        y_true_array,
        score_array,
    )

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be greater than 0 and at most 1")

    candidates: list[ThresholdSelection] = []

    for threshold in _candidate_thresholds(score_array):
        predictions = np.where(
            score_array >= threshold,
            "phishing",
            "normal",
        )

        precision = precision_score(
            y_true_array,
            predictions,
            pos_label="phishing",
            zero_division=0,
        )
        recall = recall_score(
            y_true_array,
            predictions,
            pos_label="phishing",
            zero_division=0,
        )
        f2 = fbeta_score(
            y_true_array,
            predictions,
            beta=2,
            pos_label="phishing",
            zero_division=0,
        )

        false_negative_count = int(
            ((y_true_array == "phishing") & (predictions == "normal")).sum()
        )

        candidates.append(
            ThresholdSelection(
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
                f2=float(f2),
                false_negative_count=false_negative_count,
                target_recall=target_recall,
                target_recall_met=bool(recall >= target_recall),
            )
        )

    recall_candidates = [
        candidate for candidate in candidates if candidate.target_recall_met
    ]

    selection_pool = recall_candidates if recall_candidates else candidates

    return max(
        selection_pool,
        key=lambda candidate: (
            candidate.f2,
            candidate.precision,
            candidate.threshold,
        ),
    )


def _validate_probability_inputs(
    probabilities: np.ndarray,
    labels,
    target_recall: float,
) -> np.ndarray:
    """확률 입력을 검사하고 이진 label 배열로 변환"""
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    if len(probabilities) == 0:
        raise ValueError("validation data must not be empty")

    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1")

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be between 0 and 1")

    normalized_labels = np.asarray(labels, dtype=str)
    supported_labels = {"normal", "phishing"}

    if set(normalized_labels) != supported_labels:
        raise ValueError("validation labels must contain normal and phishing")

    return (normalized_labels == "phishing").astype(int)


def _recall_and_f2(
    binary_labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float]:
    """이진 예측의 recall과 F2를 함께 계산"""
    return (
        float(recall_score(binary_labels, predictions, zero_division=0)),
        float(fbeta_score(binary_labels, predictions, beta=2, zero_division=0)),
    )


def select_probability_threshold(
    probabilities,
    labels,
    *,
    target_recall: float,
) -> ProbabilityThresholdSelection:
    """Recall 목표를 만족하는 후보 중 F2가 가장 높은 확률 임계값을 선택"""
    probability_array = np.asarray(probabilities, dtype=np.float64)
    binary_labels = _validate_probability_inputs(
        probability_array,
        labels,
        target_recall,
    )

    candidates = np.unique(
        np.concatenate([PROBABILITY_CANDIDATE_GRID, probability_array])
    )

    best: tuple[float, float, float] | None = None

    for threshold in candidates:
        predictions = (probability_array >= threshold).astype(int)
        recall, f2 = _recall_and_f2(binary_labels, predictions)

        if recall < target_recall:
            continue

        candidate = (f2, recall, float(threshold))

        if best is None or candidate > best:
            best = candidate

    if best is None:
        threshold = float(np.min(candidates))
        recall, f2 = _recall_and_f2(
            binary_labels,
            (probability_array >= threshold).astype(int),
        )

        return ProbabilityThresholdSelection(
            threshold=threshold,
            recall=recall,
            f2=f2,
            target_recall=target_recall,
            target_recall_met=False,
        )

    f2, recall, threshold = best

    return ProbabilityThresholdSelection(
        threshold=threshold,
        recall=recall,
        f2=f2,
        target_recall=target_recall,
        target_recall_met=True,
    )
