"""validation 데이터 기반 threshold 선택"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
)


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


def _validate_binary_inputs(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> None:

    """label과 score 입력의 기본 무결성을 검사"""
    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            "y_true and scores must be one-dimensional"
        )

    if len(y_true) != len(scores):
        raise ValueError(
            "y_true and scores must have the same length"
        )

    if len(y_true) == 0:
        raise ValueError(
            "cannot select threshold from empty inputs"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "scores must contain only finite numbers"
        )

    allowed_labels = {"normal", "phishing"}

    if not set(y_true).issubset(allowed_labels):
        raise ValueError(
            f"unsupported labels: {set(y_true) - allowed_labels}"
        )


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
        raise ValueError(
            "target_recall must be greater than 0 and at most 1"
        )

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
            (
                (y_true_array == "phishing")
                & (predictions == "normal")
            ).sum()
        )

        candidates.append(
            ThresholdSelection(
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
                f2=float(f2),
                false_negative_count=false_negative_count,
                target_recall=target_recall,
                target_recall_met=bool(
                    recall >= target_recall
                ),
            )
        )

    recall_candidates = [
        candidate
        for candidate in candidates
        if candidate.target_recall_met
    ]

    selection_pool = (
        recall_candidates
        if recall_candidates
        else candidates
    )

    return max(
        selection_pool,
        key=lambda candidate: (
            candidate.f2,
            candidate.precision,
            candidate.threshold,
        ),
    )