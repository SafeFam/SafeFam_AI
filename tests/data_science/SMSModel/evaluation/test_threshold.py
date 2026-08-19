"""Validation threshold 선택 정책 테스트."""

import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.evaluation.threshold import (
    ThresholdInfeasibleError,
    select_probability_threshold,
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


def test_selects_probability_threshold_using_f2_and_recall() -> None:
    """Recall 목표를 만족하는 후보 중 F2가 가장 높은 지점을 고른다"""
    selection = select_probability_threshold(
        np.asarray([0.1, 0.4, 0.8, 0.9]),
        pd.Series(["normal", "normal", "phishing", "phishing"]),
        target_recall=1.0,
    )

    assert 0.4 < selection.threshold <= 0.8
    assert selection.recall == pytest.approx(1.0)
    assert selection.f2 == pytest.approx(1.0)
    assert selection.target_recall_met is True


def test_probability_selection_exposes_validation_metrics() -> None:
    """artifact metadata에 기록하는 항목만 정확히 내보낸다"""
    selection = select_probability_threshold(
        np.asarray([0.1, 0.9]),
        pd.Series(["normal", "phishing"]),
        target_recall=1.0,
    )

    assert set(selection.to_validation_metrics()) == {
        "recall",
        "f2",
        "false_positive_rate",
        "target_recall",
        "target_recall_met",
        "max_false_positive_rate",
        "measurable_false_positive_rate",
    }


def test_probability_target_is_always_reachable() -> None:
    """후보에 관측 확률이 모두 들어 있어 Recall 목표는 언제나 달성된다

    threshold를 최저 확률까지 내리면 전부 피싱으로 판정돼 Recall이 1.0이
    된다. 그래서 현재 구조에서는 미달 분기로 갈 수 없다. FPR 상한이 들어가는
    #85 후속 단계에서야 달성 불가 상황이 실제로 생긴다.
    """
    # 피싱 확률이 정상보다 낮아 분리가 전혀 안 되는 최악의 경우
    selection = select_probability_threshold(
        np.asarray([0.9, 0.8, 0.2, 0.1]),
        pd.Series(["normal", "normal", "phishing", "phishing"]),
        target_recall=1.0,
    )

    assert selection.target_recall_met is True
    assert selection.recall == pytest.approx(1.0)


def test_two_selectors_agree_on_separable_scores() -> None:
    """정책이 갈리는 건 F2 동점 처리와 미달 시 동작뿐이다

    후보 격자와 반환 형식이 달라 보이지만, 잘 갈리는 입력에서는 같은 지점을
    고른다. 차이가 드러나는 곳을 좁혀 두어야 #85 후속 단계에서 제약을 어디에
    넣을지 판단할 수 있다.
    """
    probabilities = np.asarray([0.2, 0.3, 0.7, 0.8])
    labels = pd.Series(["normal", "normal", "phishing", "phishing"])

    probability_selection = select_probability_threshold(
        probabilities,
        labels,
        target_recall=1.0,
    )
    score_selection = select_validation_threshold(
        labels.to_numpy(),
        probabilities,
        target_recall=1.0,
    )

    assert probability_selection.threshold == pytest.approx(
        score_selection.threshold
    )
    assert probability_selection.f2 == pytest.approx(score_selection.f2)


def test_probability_selection_is_deterministic() -> None:
    """같은 입력은 같은 임계값을 준다"""
    arguments = (
        np.asarray([0.2, 0.35, 0.7, 0.95]),
        pd.Series(["normal", "phishing", "normal", "phishing"]),
    )

    assert select_probability_threshold(
        *arguments, target_recall=0.95
    ) == select_probability_threshold(*arguments, target_recall=0.95)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        (np.asarray([]), pd.Series(dtype=str), "must not be empty"),
        (
            np.asarray([0.1, np.nan]),
            pd.Series(["normal", "phishing"]),
            "finite",
        ),
        (
            np.asarray([0.1, 1.1]),
            pd.Series(["normal", "phishing"]),
            "between 0 and 1",
        ),
        (np.asarray([0.1]), pd.Series(["normal", "phishing"]), "same length"),
        (
            np.asarray([0.1, 0.9]),
            pd.Series(["normal", "normal"]),
            "must contain normal and phishing",
        ),
    ],
)
def test_probability_selection_rejects_invalid_inputs(
    probabilities: np.ndarray,
    labels: pd.Series,
    message: str,
) -> None:
    """잘못된 입력은 즉시 실패해야 한다"""
    with pytest.raises(ValueError, match=message):
        select_probability_threshold(
            probabilities,
            labels,
            target_recall=0.95,
        )


def build_ceiling_case() -> tuple[np.ndarray, pd.Series]:
    """정상 10건 중 상위 몇 건이 피싱 구간에 섞여 있는 표본"""
    probabilities = np.asarray(
        [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.55, 0.70, 0.85]
        + [0.30, 0.45, 0.60, 0.75, 0.90]
    )
    labels = pd.Series(["normal"] * 10 + ["phishing"] * 5)
    return probabilities, labels


def test_ceiling_excludes_high_false_positive_candidates() -> None:
    """상한을 넘는 후보는 F2가 높아도 선택되지 않는다"""
    probabilities, labels = build_ceiling_case()

    unconstrained = select_probability_threshold(
        probabilities, labels, target_recall=0.4
    )
    constrained = select_probability_threshold(
        probabilities,
        labels,
        target_recall=0.4,
        max_false_positive_rate=0.1,
    )

    assert unconstrained.false_positive_rate > 0.1
    assert constrained.false_positive_rate <= 0.1
    assert constrained.threshold > unconstrained.threshold


def test_raises_when_both_targets_cannot_be_met() -> None:
    """조용히 물러나지 않고 실패해야 미달이 artifact에 숨지 않는다"""
    probabilities, labels = build_ceiling_case()

    with pytest.raises(ThresholdInfeasibleError) as raised:
        select_probability_threshold(
            probabilities,
            labels,
            target_recall=1.0,
            max_false_positive_rate=0.0,
        )

    error = raised.value
    assert error.target_recall == 1.0
    assert error.max_false_positive_rate == 0.0
    # 상한 안에서 낼 수 있는 최대 Recall과, 목표 Recall의 오탐 대가를 함께 남긴다
    assert 0.0 <= error.best_recall_within_ceiling < 1.0
    assert error.lowest_false_positive_rate_at_target_recall > 0.0


def test_reports_the_resolution_of_the_validation_split() -> None:
    """정상 표본 수가 곧 오탐률의 측정 하한이다"""
    probabilities, labels = build_ceiling_case()

    selection = select_probability_threshold(
        probabilities,
        labels,
        target_recall=0.6,
        max_false_positive_rate=0.2,
    )

    # 정상 10건이므로 한 건이 0.1이다. 그보다 촘촘한 상한은 검증할 수 없다.
    assert selection.measurable_false_positive_rate == pytest.approx(0.1)


def test_infeasible_error_reports_the_resolution_limit() -> None:
    """상한이 표본 해상도보다 촘촘하면 그 사실을 메시지에 남긴다"""
    probabilities, labels = build_ceiling_case()

    with pytest.raises(ThresholdInfeasibleError) as raised:
        select_probability_threshold(
            probabilities,
            labels,
            target_recall=1.0,
            max_false_positive_rate=0.01,
        )

    assert raised.value.measurable_false_positive_rate == pytest.approx(0.1)
    assert "resolve a false positive rate" in str(raised.value)


def test_ceiling_is_optional() -> None:
    """상한을 주지 않으면 기존 정책 그대로 동작한다"""
    probabilities, labels = build_ceiling_case()

    assert select_probability_threshold(
        probabilities, labels, target_recall=0.6
    ).max_false_positive_rate is None


@pytest.mark.parametrize("ceiling", [-0.1, 1.5])
def test_rejects_out_of_range_ceiling(ceiling: float) -> None:
    """확률 범위를 벗어난 상한은 거부한다"""
    probabilities, labels = build_ceiling_case()

    with pytest.raises(ValueError, match="max_false_positive_rate"):
        select_probability_threshold(
            probabilities,
            labels,
            target_recall=0.6,
            max_false_positive_rate=ceiling,
        )
