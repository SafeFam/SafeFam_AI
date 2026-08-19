"""확률 캘리브레이션 테스트"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from data_science.SMSModel.modeling.calibration import (
    CALIBRATION_METHODS,
    ProbabilityCalibrator,
    brier_score,
    expected_calibration_error,
    fit_probability_calibrator,
)


def build_underconfident_case() -> tuple[np.ndarray, np.ndarray]:
    """확률이 실제 피싱 비율보다 낮게 나오는 표본

    v3가 real_holdout에서 보인 형태다. 확률 0.4 구간의 실제 피싱 비율이
    0.7을 넘어 경계가 다른 split으로 옮겨가지 않았다.
    """
    rng = np.random.default_rng(42)
    probabilities = np.concatenate(
        [
            rng.uniform(0.00, 0.20, 40),
            rng.uniform(0.30, 0.60, 40),
            rng.uniform(0.70, 1.00, 40),
        ]
    )
    # 중간 구간 대부분이 실제로는 피싱이다.
    labels = np.array(
        ["normal"] * 36
        + ["phishing"] * 4
        + ["normal"] * 10
        + ["phishing"] * 30
        + ["normal"] * 2
        + ["phishing"] * 38
    )
    return probabilities, labels


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_never_degrades_ranking(method: str) -> None:
    """순위를 뒤집지 않으므로 ROC-AUC가 떨어지면 안 된다

    isotonic은 계단 함수라 구간을 통째로 같은 값으로 눌러 동점을 만든다.
    동점은 순위 역전이 아니므로 AUC는 유지되거나 올라간다.
    """
    probabilities, labels = build_underconfident_case()
    truth = (labels == "phishing").astype(int)

    calibrator = fit_probability_calibrator(
        probabilities, labels, method=method
    )
    calibrated = calibrator.transform(probabilities)

    assert roc_auc_score(truth, calibrated) >= roc_auc_score(
        truth, probabilities
    )


def test_sigmoid_calibration_leaves_ranking_untouched() -> None:
    """sigmoid는 순증가라 동점도 만들지 않아 AUC가 그대로다"""
    probabilities, labels = build_underconfident_case()
    truth = (labels == "phishing").astype(int)

    calibrator = fit_probability_calibrator(
        probabilities, labels, method="sigmoid"
    )
    calibrated = calibrator.transform(probabilities)

    assert roc_auc_score(truth, calibrated) == pytest.approx(
        roc_auc_score(truth, probabilities)
    )


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_reduces_calibration_error(method: str) -> None:
    """과소 확신을 보정하면 ECE가 줄어야 한다"""
    probabilities, labels = build_underconfident_case()

    calibrator = fit_probability_calibrator(
        probabilities, labels, method=method
    )
    calibrated = calibrator.transform(probabilities)

    assert expected_calibration_error(
        calibrated, labels
    ) < expected_calibration_error(probabilities, labels)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_stays_within_probability_range(method: str) -> None:
    """보정된 값도 확률이어야 한다"""
    probabilities, labels = build_underconfident_case()

    calibrator = fit_probability_calibrator(
        probabilities, labels, method=method
    )
    calibrated = calibrator.transform(np.array([0.0, 0.5, 1.0]))

    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_calibration_is_monotone(method: str) -> None:
    """입력 확률이 커지면 보정된 확률도 작아지지 않아야 한다"""
    probabilities, labels = build_underconfident_case()

    calibrator = fit_probability_calibrator(
        probabilities, labels, method=method
    )
    calibrated = calibrator.transform(np.linspace(0.0, 1.0, 50))

    assert (np.diff(calibrated) >= -1e-12).all()


def test_transform_handles_empty_input() -> None:
    """빈 배열은 그대로 빈 배열을 반환한다"""
    probabilities, labels = build_underconfident_case()
    calibrator = fit_probability_calibrator(probabilities, labels)

    assert len(calibrator.transform(np.array([]))) == 0


def test_calibrator_records_training_size() -> None:
    """metadata에 남길 정보를 함께 보관한다"""
    probabilities, labels = build_underconfident_case()

    calibrator = fit_probability_calibrator(
        probabilities, labels, method="sigmoid"
    )

    assert calibrator.to_dict() == {
        "method": "sigmoid",
        "training_sample_count": len(probabilities),
    }


def test_rejects_unknown_method() -> None:
    """지원하지 않는 방법은 조용히 넘어가지 않는다"""
    probabilities, labels = build_underconfident_case()

    with pytest.raises(ValueError, match="unsupported calibration method"):
        fit_probability_calibrator(probabilities, labels, method="beta")


def test_transform_rejects_multidimensional_input() -> None:
    """2차원 입력은 거부한다"""
    probabilities, labels = build_underconfident_case()
    calibrator = fit_probability_calibrator(probabilities, labels)

    with pytest.raises(ValueError, match="one-dimensional"):
        calibrator.transform(np.zeros((2, 2)))


def test_brier_score_is_zero_for_perfect_predictions() -> None:
    """완벽한 예측의 Brier score는 0이다"""
    probabilities = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array(["normal", "phishing", "normal", "phishing"])

    assert brier_score(probabilities, labels) == 0.0


def test_expected_calibration_error_is_zero_when_calibrated() -> None:
    """예측 확률과 실제 비율이 같으면 ECE는 0이다"""
    # 확률 0.5 구간에 정상과 피싱이 정확히 절반씩 있다.
    probabilities = np.full(10, 0.5)
    labels = np.array(["normal"] * 5 + ["phishing"] * 5)

    assert expected_calibration_error(probabilities, labels) == pytest.approx(
        0.0
    )


def test_expected_calibration_error_measures_the_gap() -> None:
    """예측이 실제보다 낮으면 그 차이가 그대로 잡힌다"""
    probabilities = np.full(10, 0.4)
    labels = np.array(["normal"] * 2 + ["phishing"] * 8)

    assert expected_calibration_error(probabilities, labels) == pytest.approx(
        0.4
    )


def test_expected_calibration_error_rejects_empty_bins() -> None:
    """구간 수는 최소 1이어야 한다"""
    probabilities, labels = build_underconfident_case()

    with pytest.raises(ValueError, match="bin_count must be at least 1"):
        expected_calibration_error(probabilities, labels, bin_count=0)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([0.1, 0.2], ["normal"], "same length"),
        ([], [], "empty inputs"),
        ([0.1, np.nan], ["normal", "phishing"], "finite"),
        ([0.1, 1.5], ["normal", "phishing"], "between 0 and 1"),
        ([0.1, 0.2], ["normal", "normal"], "both normal and phishing"),
        ([0.1, 0.2], ["normal", "spam"], "unsupported labels"),
    ],
)
def test_fit_rejects_invalid_inputs(
    probabilities: list,
    labels: list,
    message: str,
) -> None:
    """잘못된 입력은 즉시 실패해야 한다"""
    with pytest.raises(ValueError, match=message):
        fit_probability_calibrator(
            np.asarray(probabilities, dtype=float),
            np.asarray(labels, dtype=str),
        )


def test_calibrator_is_frozen() -> None:
    """학습된 캘리브레이터는 이후 변경되지 않아야 한다"""
    probabilities, labels = build_underconfident_case()
    calibrator = fit_probability_calibrator(probabilities, labels)

    with pytest.raises(AttributeError):
        calibrator.method = "sigmoid"

    assert isinstance(calibrator, ProbabilityCalibrator)
