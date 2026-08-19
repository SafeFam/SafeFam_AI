"""Stacking 확률 캘리브레이션"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ISOTONIC = "isotonic"
SIGMOID = "sigmoid"

CALIBRATION_METHODS = (ISOTONIC, SIGMOID)

LOGIT_CLIP = 1e-6


def _validate_probabilities(probabilities) -> np.ndarray:
    """확률 배열이 유한하고 0과 1 사이인지 검사

    sklearn은 비유한 입력을 조용히 흘리거나 알아보기 어려운 오류를 내고,
    범위를 벗어난 값은 clip으로 삼켜진다. 들어오는 자리에서 막는다.
    """
    probability_array = np.asarray(probabilities, dtype=np.float64)

    if probability_array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")

    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1")

    return probability_array


def _validate(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    """확률과 label 입력을 검사하고 정규화"""
    probability_array = _validate_probabilities(probabilities)
    label_array = np.asarray(labels, dtype=str)

    if label_array.ndim != 1:
        raise ValueError("probabilities and labels must be one-dimensional")

    if len(probability_array) != len(label_array):
        raise ValueError("probabilities and labels must have the same length")

    if len(probability_array) == 0:
        raise ValueError("cannot calibrate from empty inputs")

    allowed_labels = {"normal", "phishing"}
    observed_labels = set(label_array)

    if not observed_labels.issubset(allowed_labels):
        raise ValueError(
            f"unsupported labels: {observed_labels - allowed_labels}"
        )

    if observed_labels != allowed_labels:
        raise ValueError("labels must contain both normal and phishing")

    return probability_array, label_array


def _to_logit(probabilities: np.ndarray) -> np.ndarray:
    """확률을 logit으로 변환"""
    clipped = np.clip(probabilities, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """확률을 실제 피싱 비율에 맞춰 다시 매핑하는 단조 변환"""

    method: str
    model: object
    training_sample_count: int

    def transform(self, probabilities) -> np.ndarray:
        """보정된 확률 반환"""
        probability_array = _validate_probabilities(probabilities)

        if len(probability_array) == 0:
            return probability_array

        if self.method == ISOTONIC:
            calibrated = self.model.predict(probability_array)
        else:
            calibrated = self.model.predict_proba(
                _to_logit(probability_array).reshape(-1, 1)
            )[:, 1]

        return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        """metadata에 기록할 dictionary로 변환"""
        return {
            "method": self.method,
            "training_sample_count": self.training_sample_count,
        }


def fit_probability_calibrator(
    probabilities,
    labels,
    *,
    method: str = ISOTONIC,
) -> ProbabilityCalibrator:
    """보정용 split에서 캘리브레이터를 학습"""
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"unsupported calibration method: {method}. "
            f"expected one of {CALIBRATION_METHODS}"
        )

    probability_array, label_array = _validate(probabilities, labels)
    targets = (label_array == "phishing").astype(int)

    if method == ISOTONIC:
        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        model.fit(probability_array, targets)
    else:
        model = LogisticRegression()
        model.fit(_to_logit(probability_array).reshape(-1, 1), targets)

        # 계수가 음수면 확률이 클수록 낮게 보정돼 순서가 뒤집힌다.
        # 점수가 label과 역상관이라는 뜻이므로 보정할 대상이 아니다.
        if float(model.coef_[0, 0]) < 0.0:
            raise ValueError(
                "sigmoid calibration produced a decreasing mapping; "
                "the scores are inversely related to the labels"
            )

    return ProbabilityCalibrator(
        method=method,
        model=model,
        training_sample_count=len(probability_array),
    )


def brier_score(probabilities, labels) -> float:
    """예측 확률과 실제 결과의 평균 제곱 오차"""
    probability_array, label_array = _validate(probabilities, labels)
    targets = (label_array == "phishing").astype(np.float64)

    return float(np.mean((probability_array - targets) ** 2))


def expected_calibration_error(
    probabilities,
    labels,
    *,
    bin_count: int = 10,
) -> float:
    """구간별 예측 확률과 실제 비율 차이를 표본 수로 가중 평균"""
    probability_array, label_array = _validate(probabilities, labels)

    if bin_count < 1:
        raise ValueError("bin_count must be at least 1")

    targets = (label_array == "phishing").astype(np.float64)
    edges = np.linspace(0.0, 1.0, bin_count + 1)

    total_error = 0.0
    for index in range(bin_count):
        lower = edges[index]
        upper = edges[index + 1]

        if index == bin_count - 1:
            in_bin = (probability_array >= lower) & (probability_array <= upper)
        else:
            in_bin = (probability_array >= lower) & (probability_array < upper)

        if not in_bin.any():
            continue

        gap = abs(
            probability_array[in_bin].mean() - targets[in_bin].mean()
        )
        total_error += (in_bin.sum() / len(probability_array)) * gap

    return float(total_error)
