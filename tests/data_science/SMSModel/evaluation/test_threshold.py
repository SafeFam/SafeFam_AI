"""Validation threshold 선택 정책 테스트."""

import numpy as np
import pytest

from data_science.SMSModel.evaluation.threshold import (
    select_validation_threshold,
)


def test_selects_highest_quality_threshold_meeting_recall_target():
    result = select_validation_threshold(
        ["normal", "normal", "phishing", "phishing"],
        [0.1, 0.4, 0.6, 0.9],
        target_recall=1.0,
    )

    assert result.threshold == pytest.approx(0.6)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f2 == pytest.approx(1.0)
    assert result.false_negative_count == 0
    assert result.target_recall_met is True


def test_threshold_selection_is_deterministic():
    arguments = (
        ["normal", "phishing", "normal", "phishing"],
        [0.2, 0.7, 0.4, 0.8],
    )
    assert select_validation_threshold(*arguments) == select_validation_threshold(
        *arguments
    )


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1])
def test_rejects_invalid_target_recall(target):
    with pytest.raises(ValueError, match="target_recall"):
        select_validation_threshold(
            ["normal", "phishing"],
            [0.1, 0.9],
            target_recall=target,
        )


@pytest.mark.parametrize(
    "labels, scores, message",
    [
        ([], [], "empty"),
        (["normal"], [0.1, 0.2], "same length"),
        (["unknown"], [0.1], "unsupported labels"),
        (["normal"], [np.nan], "finite"),
    ],
)
def test_rejects_invalid_validation_inputs(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        select_validation_threshold(labels, scores)


@pytest.mark.parametrize(
    "labels",
    [["normal", "normal"], ["phishing", "phishing"]],
)
def test_rejects_single_class_validation_data(labels):
    with pytest.raises(ValueError, match="both normal and phishing"):
        select_validation_threshold(labels, [0.1, 0.9])
