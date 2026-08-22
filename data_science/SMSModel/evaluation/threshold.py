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


class ThresholdInfeasibleError(ValueError):
    """Recall 목표와 정상 오탐 상한을 동시에 만족하는 임계값이 없을 때"""

    def __init__(
        self,
        *,
        target_recall: float,
        max_false_positive_rate: float,
        best_recall_within_ceiling: float,
        lowest_false_positive_rate_at_target_recall: float,
        measurable_false_positive_rate: float,
    ) -> None:
        self.target_recall = target_recall
        self.max_false_positive_rate = max_false_positive_rate
        self.best_recall_within_ceiling = best_recall_within_ceiling
        self.lowest_false_positive_rate_at_target_recall = (
            lowest_false_positive_rate_at_target_recall
        )
        self.measurable_false_positive_rate = measurable_false_positive_rate

        super().__init__(
            "no threshold satisfies both targets: "
            f"recall >= {target_recall} needs a false positive rate of "
            f"{lowest_false_positive_rate_at_target_recall:.4f}, while a "
            f"ceiling of {max_false_positive_rate} caps recall at "
            f"{best_recall_within_ceiling:.4f}. The validation split can only "
            f"resolve a false positive rate down to "
            f"{measurable_false_positive_rate:.4f}."
        )


@dataclass(frozen=True)
class ProbabilityThresholdSelection:
    """확률 출력 모델에서 선택된 threshold와 validation 지표"""

    threshold: float
    recall: float
    f2: float
    false_positive_rate: float
    target_recall: float
    target_recall_met: bool
    max_false_positive_rate: float | None
    measurable_false_positive_rate: float

    def to_validation_metrics(self) -> dict[str, float]:
        """artifact metadata에 기록하는 validation 항목으로 변환"""
        return {
            "recall": self.recall,
            "f2": self.f2,
            "false_positive_rate": self.false_positive_rate,
            "target_recall": self.target_recall,
            "target_recall_met": self.target_recall_met,
            "max_false_positive_rate": self.max_false_positive_rate,
            "measurable_false_positive_rate": (
                self.measurable_false_positive_rate
            ),
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
    max_false_positive_rate: float | None = None,
) -> ProbabilityThresholdSelection:
    """Recall 하한과 정상 오탐 상한을 함께 만족하는 임계값을 선택"""
    probability_array = np.asarray(probabilities, dtype=np.float64)
    binary_labels = _validate_probability_inputs(
        probability_array,
        labels,
        target_recall,
    )

    if max_false_positive_rate is not None and not (
        0.0 <= max_false_positive_rate <= 1.0
    ):
        raise ValueError(
            "max_false_positive_rate must be between 0 and 1"
        )

    is_normal = binary_labels == 0
    normal_count = int(is_normal.sum())

    if normal_count == 0:
        raise ValueError("validation data must contain normal messages")

    measurable_false_positive_rate = 1.0 / normal_count

    within_ceiling: list[tuple[float, float, float, float]] = []
    meeting_recall_rates: list[float] = []

    for threshold in _candidate_thresholds(probability_array):
        predictions = (probability_array >= threshold).astype(int)
        recall, f2 = _recall_and_f2(binary_labels, predictions)
        false_positive_rate = float(
            (predictions[is_normal] == 1).sum() / normal_count
        )

        if recall >= target_recall:
            meeting_recall_rates.append(false_positive_rate)

        if (
            max_false_positive_rate is not None
            and false_positive_rate > max_false_positive_rate
        ):
            continue

        within_ceiling.append(
            (f2, recall, float(threshold), false_positive_rate)
        )

    feasible = [
        candidate
        for candidate in within_ceiling
        if candidate[1] >= target_recall
    ]

    if not feasible:
        raise ThresholdInfeasibleError(
            target_recall=target_recall,
            max_false_positive_rate=float(max_false_positive_rate),
            best_recall_within_ceiling=max(
                (candidate[1] for candidate in within_ceiling),
                default=0.0,
            ),
            lowest_false_positive_rate_at_target_recall=min(
                meeting_recall_rates,
                default=1.0,
            ),
            measurable_false_positive_rate=measurable_false_positive_rate,
        )

    f2, recall, threshold, false_positive_rate = max(feasible)

    return ProbabilityThresholdSelection(
        threshold=threshold,
        recall=recall,
        f2=f2,
        false_positive_rate=false_positive_rate,
        target_recall=target_recall,
        target_recall_met=True,
        max_false_positive_rate=max_false_positive_rate,
        measurable_false_positive_rate=measurable_false_positive_rate,
    )


def select_probability_threshold_with_fallback(
    probabilities,
    labels,
    *,
    target_recall: float,
    max_false_positive_rate: float,
) -> tuple[ProbabilityThresholdSelection, bool]:
    """상한을 만족하는 threshold가 없으면 실제로 달성 가능한 최소 오탐률로
    자동 완화해 재시도한다.

    고정된 오탐 상한은 validation 구성이 바뀔 때마다 목표 recall에 필요한
    오탐률과 어긋나기 쉬워, 매번 사람이 상한 값을 추측해 재실행해야 했다.
    이 함수는 그 수작업을 없앤다. target_recall 자체가 이 validation에서
    달성 불가능한 경우(완화로도 못 고치는 경우)에는 원래 예외를 그대로
    전파한다 - 이건 상한 조정이 아니라 모델/데이터 문제이기 때문이다.

    Returns:
        (선택된 threshold, 상한을 완화했는지 여부)
    """
    try:
        return (
            select_probability_threshold(
                probabilities,
                labels,
                target_recall=target_recall,
                max_false_positive_rate=max_false_positive_rate,
            ),
            False,
        )
    except ThresholdInfeasibleError as error:
        relaxed_ceiling = error.lowest_false_positive_rate_at_target_recall
        if relaxed_ceiling >= 1.0:
            raise
        return (
            select_probability_threshold(
                probabilities,
                labels,
                target_recall=target_recall,
                max_false_positive_rate=relaxed_ceiling,
            ),
            True,
        )
