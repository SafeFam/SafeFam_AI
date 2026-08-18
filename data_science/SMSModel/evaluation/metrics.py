"""SMS 피싱 모델 공통 평가 지표"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

LABEL_ORDER = ["normal", "phishing"]


@dataclass(frozen=True)
class ClassificationMetrics:
    """피싱 클래스를 positive로 사용하는 이진 분류 평가 결과"""

    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 직렬화 가능한 딕셔너리를 반환"""
        return asdict(self)


def calculate_classification_metrics(
    y_true,
    y_pred,
) -> ClassificationMetrics:
    """Accuracy, Precision, Recall, F1, F2와 혼동행렬을 계산"""

    y_true_array = np.asarray(y_true, dtype=str)
    y_pred_array = np.asarray(y_pred, dtype=str)

    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional"
        )

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred must have the same length"
        )

    if len(y_true_array) == 0:
        raise ValueError("cannot evaluate empty inputs")

    allowed_labels = set(LABEL_ORDER)

    if not set(y_true_array).issubset(allowed_labels):
        raise ValueError("y_true contains unsupported labels")

    if not set(y_pred_array).issubset(allowed_labels):
        raise ValueError("y_pred contains unsupported labels")

    matrix = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=LABEL_ORDER,
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    return ClassificationMetrics(
        sample_count=len(y_true_array),
        accuracy=float(
            accuracy_score(y_true_array, y_pred_array)
        ),
        precision=float(
            precision_score(
                y_true_array,
                y_pred_array,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true_array,
                y_pred_array,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true_array,
                y_pred_array,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        f2=float(
            fbeta_score(
                y_true_array,
                y_pred_array,
                beta=2,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )