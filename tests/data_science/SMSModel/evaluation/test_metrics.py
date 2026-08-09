"""분류 성능 및 혼동행렬 테스트."""

import pytest

from data_science.SMSModel.evaluation.metrics import (
    calculate_classification_metrics,
)


def test_calculates_metrics_and_confusion_matrix_in_fixed_order():
    metrics = calculate_classification_metrics(
        ["normal", "normal", "phishing", "phishing"],
        ["normal", "phishing", "normal", "phishing"],
    )

    assert metrics.sample_count == 4
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.f2 == pytest.approx(0.5)
    assert metrics.to_dict()["false_negative"] == 1


def test_handles_zero_positive_predictions_without_division_error():
    metrics = calculate_classification_metrics(
        ["normal", "phishing"],
        ["normal", "normal"],
    )
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.false_negative == 1


@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        ([], [], "empty"),
        (["normal"], ["normal", "phishing"], "same length"),
        (["unknown"], ["normal"], "y_true"),
        (["normal"], ["unknown"], "y_pred"),
    ],
)
def test_rejects_invalid_metric_inputs(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        calculate_classification_metrics(y_true, y_pred)
