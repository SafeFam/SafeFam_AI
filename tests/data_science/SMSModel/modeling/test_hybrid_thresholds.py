"""하이브리드 조건부 호출 임계값 선정 테스트"""

import numpy as np
import pytest

from data_science.SMSModel.modeling.hybrid_thresholds import (
    select_hybrid_thresholds,
)


def test_selects_thresholds_using_recall_f2_and_call_rate():
    labels = np.asarray(
        [
            "normal",
            "normal",
            "normal",
            "phishing",
            "phishing",
            "phishing",
        ]
    )

    stacking_probabilities = np.asarray(
        [
            0.05,
            0.10,
            0.45,
            0.55,
            0.85,
            0.95,
        ]
    )

    gemini_scores = np.asarray(
        [
            5,
            10,
            15,
            85,
            90,
            95,
        ]
    )

    result = select_hybrid_thresholds(
        stacking_probabilities=(
            stacking_probabilities
        ),
        gemini_scores=gemini_scores,
        labels=labels,
        target_recall=1.0,
    )

    assert (
        result.normal_probability_max
        < result.phishing_probability_min
    )
    assert result.recall == pytest.approx(1.0)
    assert result.f2 == pytest.approx(1.0)
    assert 0.0 <= result.gemini_call_rate <= 1.0


def test_prefers_lower_gemini_call_rate_when_f2_is_equal():
    labels = np.asarray(
        [
            "normal",
            "normal",
            "phishing",
            "phishing",
        ]
    )

    stacking_probabilities = np.asarray(
        [
            0.05,
            0.10,
            0.90,
            0.95,
        ]
    )

    gemini_scores = np.asarray(
        [
            5,
            10,
            90,
            95,
        ]
    )

    result = select_hybrid_thresholds(
        stacking_probabilities=(
            stacking_probabilities
        ),
        gemini_scores=gemini_scores,
        labels=labels,
        target_recall=1.0,
    )

    # Stacking만으로 완벽히 분류할 수 있으므로 Gemini 호출이 필요하지 않아야 함
    assert result.gemini_call_count == 0
    assert result.gemini_call_rate == pytest.approx(0.0)


@pytest.mark.parametrize(
    (
        "stacking_probabilities",
        "gemini_scores",
        "labels",
    ),
    [
        (
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
        ),
        (
            np.asarray([0.1, 0.9]),
            np.asarray([10]),
            np.asarray(
                ["normal", "phishing"]
            ),
        ),
        (
            np.asarray([0.1, 1.1]),
            np.asarray([10, 90]),
            np.asarray(
                ["normal", "phishing"]
            ),
        ),
        (
            np.asarray([0.1, 0.9]),
            np.asarray([10, 101]),
            np.asarray(
                ["normal", "phishing"]
            ),
        ),
    ],
)
def test_rejects_invalid_inputs(
    stacking_probabilities,
    gemini_scores,
    labels,
):
    with pytest.raises(ValueError):
        select_hybrid_thresholds(
            stacking_probabilities=(
                stacking_probabilities
            ),
            gemini_scores=gemini_scores,
            labels=labels,
        )