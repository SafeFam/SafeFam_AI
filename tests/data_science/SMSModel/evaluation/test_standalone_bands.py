"""단독 운영 구간 측정 테스트"""

import numpy as np
import pytest

from data_science.SMSModel.evaluation.standalone_bands import (
    CERTAIN_NORMAL,
    CERTAIN_PHISHING,
    UNCERTAIN,
    BandEdges,
    BandEdgesUnreachableError,
    assign_bands,
    measure_edge_transfer,
    measure_reliability,
    select_standalone_bands,
    summarize_bands,
    sweep_band_frontier,
)


def build_separable_case() -> tuple[np.ndarray, np.ndarray]:
    """정상과 피싱이 확률로 잘 갈리는 표본"""
    probabilities = np.array(
        [0.02, 0.05, 0.08, 0.11, 0.40, 0.60, 0.92, 0.95, 0.97, 0.99]
    )
    labels = np.array(
        ["normal"] * 4 + ["normal", "phishing"] + ["phishing"] * 4
    )
    return probabilities, labels


def test_assign_bands_places_boundaries_in_outer_bands() -> None:
    """경계값은 '의심'이 아니라 바깥쪽 구간에 들어가야 한다"""
    edges = BandEdges(normal_max=0.2, phishing_min=0.8)

    bands = assign_bands(np.array([0.1, 0.2, 0.5, 0.8, 0.9]), edges)

    assert list(bands) == [
        CERTAIN_NORMAL,
        CERTAIN_NORMAL,
        UNCERTAIN,
        CERTAIN_PHISHING,
        CERTAIN_PHISHING,
    ]


def test_assign_bands_prefers_alert_when_edges_touch() -> None:
    """두 경계가 같으면 자동 경고가 우선한다"""
    edges = BandEdges(normal_max=0.5, phishing_min=0.5)

    assert list(assign_bands(np.array([0.5]), edges)) == [CERTAIN_PHISHING]


def test_band_edges_reject_crossed_boundaries() -> None:
    """무알림 경계가 자동 경고 경계보다 높으면 3구간이 성립하지 않는다"""
    with pytest.raises(ValueError, match="normal_max must not exceed"):
        BandEdges(normal_max=0.9, phishing_min=0.1)


@pytest.mark.parametrize(
    ("normal_max", "phishing_min"),
    [(-0.1, 0.5), (0.5, 1.2)],
)
def test_band_edges_reject_out_of_range(
    normal_max: float,
    phishing_min: float,
) -> None:
    """확률 범위를 벗어난 경계는 거부한다"""
    with pytest.raises(ValueError, match="between 0 and 1"):
        BandEdges(normal_max=normal_max, phishing_min=phishing_min)


def test_summarize_bands_reports_operating_errors() -> None:
    """자동 경고 오탐과 놓친 피싱을 각각 집계한다"""
    probabilities, labels = build_separable_case()
    edges = BandEdges(normal_max=0.10, phishing_min=0.90)

    summary = summarize_bands(probabilities, labels, edges)

    # 정상 4건 중 0.11은 '의심'으로 올라간다.
    assert summary["by_band"][CERTAIN_NORMAL]["normal_count"] == 3
    assert summary["by_band"][UNCERTAIN]["sample_count"] == 3
    assert summary["by_band"][CERTAIN_PHISHING]["phishing_count"] == 4

    # 자동 경고 구간에 정상이 없으므로 오경보는 0이다.
    assert summary["alert_false_positive_rate"] == 0.0
    assert summary["alert_recall"] == pytest.approx(4 / 5)
    # 피싱 5건 모두 최소 '의심' 이상에 있다.
    assert summary["coverage_recall"] == 1.0
    assert summary["missed_phishing_rate"] == 0.0


def test_summarize_bands_counts_missed_phishing() -> None:
    """무알림 구간으로 빠진 피싱이 놓침으로 잡혀야 한다"""
    probabilities, labels = build_separable_case()
    edges = BandEdges(normal_max=0.70, phishing_min=0.90)

    summary = summarize_bands(probabilities, labels, edges)

    # 0.60 피싱 한 건이 무알림으로 내려간다.
    assert summary["missed_phishing_rate"] == pytest.approx(1 / 5)
    assert summary["coverage_recall"] == pytest.approx(4 / 5)


def test_summarize_bands_matches_single_threshold_when_edges_touch() -> None:
    """경계를 붙이면 단일 임계값 결과와 같아야 한다"""
    probabilities, labels = build_separable_case()
    threshold = 0.5
    edges = BandEdges(
        normal_max=float(np.nextafter(threshold, -np.inf)),
        phishing_min=threshold,
    )

    summary = summarize_bands(probabilities, labels, edges)

    predicted = probabilities >= threshold
    is_phishing = labels == "phishing"

    assert summary["uncertain_share"] == 0.0
    assert summary["alert_recall"] == pytest.approx(
        (predicted & is_phishing).sum() / is_phishing.sum()
    )
    assert summary["alert_false_positive_rate"] == pytest.approx(
        (predicted & ~is_phishing).sum() / (~is_phishing).sum()
    )


def test_fully_overlapping_scores_push_everything_into_uncertain() -> None:
    """구분이 전혀 안 되면 두 목표는 지켜지되 표본 전부가 '의심'이 된다"""
    # 정상과 피싱이 같은 확률이라 어떤 경계로도 갈라지지 않는다.
    probabilities = np.array([0.5] * 8)
    labels = np.array(["normal"] * 4 + ["phishing"] * 4)

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.0,),
        coverage_recall_targets=(1.0,),
    )

    assert len(frontier) == 1
    measured = frontier[0]["measured"]

    # 목표 자체는 지켜진다. 자동 경고도 무알림도 비어 있기 때문이다.
    assert measured["alert_false_positive_rate"] == 0.0
    assert measured["coverage_recall"] == 1.0
    # 달성 여부가 아니라 이 값이 모델이 쓸모없음을 드러낸다.
    assert measured["uncertain_share"] == 1.0


def test_sweep_reports_unreachable_alert_ceiling() -> None:
    """정상이 확률 1.0에 있으면 오탐 0건 상한을 만족할 수 없다"""
    probabilities = np.array([0.1, 1.0, 0.9])
    labels = np.array(["normal", "normal", "phishing"])

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.0,),
        coverage_recall_targets=(1.0,),
    )

    assert frontier[0]["feasible"] is False
    assert frontier[0]["reason"] == "ALERT_CEILING_UNREACHABLE"


def test_sweep_reports_unreachable_coverage_target() -> None:
    """피싱이 확률 0.0에 있으면 전부 포착하는 경계를 둘 수 없다"""
    probabilities = np.array([0.1, 0.0, 0.9])
    labels = np.array(["normal", "phishing", "phishing"])

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.5,),
        coverage_recall_targets=(1.0,),
    )

    assert frontier[0]["feasible"] is False
    assert frontier[0]["reason"] == "COVERAGE_TARGET_UNREACHABLE"


def test_sweep_respects_the_alert_false_positive_ceiling() -> None:
    """자동 경고 구간의 정상 오탐이 상한을 넘지 않아야 한다"""
    probabilities, labels = build_separable_case()

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.2,),
        coverage_recall_targets=(0.8,),
    )

    measured = frontier[0]["measured"]
    assert measured["alert_false_positive_rate"] <= 0.2
    assert measured["coverage_recall"] >= 0.8


def test_sweep_covers_every_target_combination() -> None:
    """목표 조합마다 한 줄씩 남겨야 비교표를 만들 수 있다"""
    probabilities, labels = build_separable_case()

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.0, 0.25),
        coverage_recall_targets=(0.8, 1.0),
    )

    assert len(frontier) == 4
    assert {
        (
            entry["target_alert_false_positive_rate"],
            entry["target_coverage_recall"],
        )
        for entry in frontier
    } == {(0.0, 0.8), (0.0, 1.0), (0.25, 0.8), (0.25, 1.0)}


def test_zero_ceiling_keeps_every_normal_out_of_the_alert_band() -> None:
    """오탐 0건 상한이면 자동 경고에 정상이 한 건도 없어야 한다"""
    probabilities, labels = build_separable_case()

    frontier = sweep_band_frontier(
        probabilities,
        labels,
        alert_false_positive_targets=(0.0,),
        coverage_recall_targets=(0.8,),
    )

    measured = frontier[0]["measured"]
    assert measured["by_band"][CERTAIN_PHISHING]["normal_count"] == 0


def test_measure_reliability_bins_observed_rates() -> None:
    """구간별 예측 확률과 실제 피싱 비율을 함께 보고한다"""
    probabilities, labels = build_separable_case()

    bins = measure_reliability(
        probabilities,
        labels,
        bin_edges=(0.0, 0.5, 1.0),
    )

    assert len(bins) == 2
    # 0.5 아래는 정상 5건이다.
    assert bins[0]["sample_count"] == 5
    assert bins[0]["observed_phishing_rate"] == 0.0
    # 0.5 위는 피싱 5건이다.
    assert bins[1]["sample_count"] == 5
    assert bins[1]["observed_phishing_rate"] == 1.0
    assert bins[1]["mean_predicted_probability"] == pytest.approx(
        probabilities[5:].mean()
    )


def test_measure_reliability_includes_the_upper_bound() -> None:
    """확률 1.0이 마지막 구간에서 누락되지 않아야 한다"""
    probabilities = np.array([0.0, 1.0])
    labels = np.array(["normal", "phishing"])

    bins = measure_reliability(probabilities, labels, bin_edges=(0.0, 0.5, 1.0))

    assert sum(bin_result["sample_count"] for bin_result in bins) == 2


def test_measure_reliability_rejects_unsorted_edges() -> None:
    """구간 경계가 증가하지 않으면 거부한다"""
    probabilities, labels = build_separable_case()

    with pytest.raises(ValueError, match="strictly increasing"):
        measure_reliability(probabilities, labels, bin_edges=(0.0, 0.5, 0.3))


def test_measure_edge_transfer_reports_drift_against_selection_split() -> None:
    """선정 split 대비 다른 split의 지표 변화를 기록한다"""
    probabilities, labels = build_separable_case()
    edges = BandEdges(normal_max=0.10, phishing_min=0.90)

    # 자동 경고 구간에 정상 한 건이 들어오도록 확률을 밀어올린 split
    shifted = probabilities.copy()
    shifted[0] = 0.95

    transfers = measure_edge_transfer(
        edges,
        "validation",
        {
            "validation": (probabilities, labels),
            "real_holdout": (shifted, labels),
        },
    )

    selection = next(t for t in transfers if t["split"] == "validation")
    assert selection["is_selection_split"] is True
    assert selection["alert_false_positive_rate_drift"] == 0.0

    holdout = next(t for t in transfers if t["split"] == "real_holdout")
    assert holdout["alert_false_positive_rate"] == pytest.approx(1 / 5)
    assert holdout["alert_false_positive_rate_drift"] == pytest.approx(1 / 5)


def test_measure_edge_transfer_requires_the_selection_split() -> None:
    """선정 split이 측정 대상에 없으면 기준선을 만들 수 없다"""
    probabilities, labels = build_separable_case()

    with pytest.raises(ValueError, match="selection split not measured"):
        measure_edge_transfer(
            BandEdges(normal_max=0.1, phishing_min=0.9),
            "validation",
            {"test": (probabilities, labels)},
        )


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
def test_summarize_bands_rejects_invalid_inputs(
    probabilities: list,
    labels: list,
    message: str,
) -> None:
    """잘못된 입력은 조용히 넘어가지 않고 즉시 실패해야 한다"""
    with pytest.raises(ValueError, match=message):
        summarize_bands(
            np.asarray(probabilities, dtype=float),
            np.asarray(labels, dtype=str),
            BandEdges(normal_max=0.1, phishing_min=0.9),
        )


def test_selects_bands_from_probabilities_alone() -> None:
    """LLM 점수 없이 확률만으로 두 목표를 지키는 경계를 고른다"""
    probabilities, labels = build_separable_case()

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=0.0,
        min_coverage_recall=1.0,
    )
    measured = summarize_bands(probabilities, labels, edges)

    assert measured["alert_false_positive_rate"] == 0.0
    assert measured["coverage_recall"] == 1.0


def test_selection_clamps_the_silent_boundary() -> None:
    """무알림 경계가 자동 경고 경계를 넘으면 거기서 잘라낸다"""
    probabilities, labels = build_separable_case()

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=0.2,
        min_coverage_recall=0.8,
    )

    assert edges.normal_max <= edges.phishing_min
    # 잘라내도 두 목표는 그대로 지켜진다.
    measured = summarize_bands(probabilities, labels, edges)
    assert measured["alert_false_positive_rate"] <= 0.2
    assert measured["coverage_recall"] >= 0.8


def test_selection_raises_when_the_alert_boundary_cannot_be_placed() -> None:
    """정상이 확률 1.0에 있으면 오탐 0건 경계를 놓을 수 없다"""
    with pytest.raises(BandEdgesUnreachableError) as raised:
        select_standalone_bands(
            np.array([0.1, 1.0, 0.9]),
            np.array(["normal", "normal", "phishing"]),
            max_alert_false_positive_rate=0.0,
            min_coverage_recall=1.0,
        )

    assert raised.value.reason == "ALERT_CEILING_UNREACHABLE"


def test_selection_raises_when_the_silent_boundary_cannot_be_placed() -> None:
    """피싱이 확률 0.0에 있으면 전부 포착하는 경계를 놓을 수 없다"""
    with pytest.raises(BandEdgesUnreachableError) as raised:
        select_standalone_bands(
            np.array([0.1, 0.0, 0.9]),
            np.array(["normal", "phishing", "phishing"]),
            max_alert_false_positive_rate=0.5,
            min_coverage_recall=1.0,
        )

    assert raised.value.reason == "COVERAGE_TARGET_UNREACHABLE"


@pytest.mark.parametrize(
    ("alert_ceiling", "coverage_floor"),
    [(-0.1, 0.95), (1.5, 0.95), (0.01, 0.0), (0.01, 1.5)],
)
def test_selection_rejects_out_of_range_targets(
    alert_ceiling: float,
    coverage_floor: float,
) -> None:
    """범위를 벗어난 목표값은 거부한다"""
    probabilities, labels = build_separable_case()

    with pytest.raises(ValueError, match="must be"):
        select_standalone_bands(
            probabilities,
            labels,
            max_alert_false_positive_rate=alert_ceiling,
            min_coverage_recall=coverage_floor,
        )


def build_tied_case() -> tuple[np.ndarray, np.ndarray]:
    """정상 여러 건이 같은 확률에 묶여 있는 표본"""
    probabilities = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.6, 0.95])
    labels = np.array(["normal"] * 5 + ["phishing"] * 2)
    return probabilities, labels


def test_tied_probabilities_do_not_break_the_alert_ceiling() -> None:
    """경계값에 동점이 몰려도 오탐 상한을 넘기지 않아야 함"""
    probabilities, labels = build_tied_case()

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=0.4,
        min_coverage_recall=1.0,
    )
    measured = summarize_bands(probabilities, labels, edges)

    assert measured["alert_false_positive_rate"] <= 0.4


def test_tied_probabilities_do_not_break_the_coverage_floor() -> None:
    """무알림 경계도 동점 때문에 놓침 예산을 넘기면 안됨"""
    probabilities = np.array([0.1, 0.4, 0.4, 0.4, 0.9, 0.05, 0.2])
    labels = np.array(["phishing"] * 5 + ["normal"] * 2)

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=0.5,
        min_coverage_recall=0.8,
    )
    measured = summarize_bands(probabilities, labels, edges)

    assert measured["coverage_recall"] >= 0.8


def test_untied_selection_is_unchanged() -> None:
    """동점이 없으면 예산을 정확히 소진하는 기존 동작 그대로"""
    probabilities = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.95, 0.99])
    labels = np.array(["normal"] * 5 + ["phishing"] * 2)

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=0.4,
        min_coverage_recall=1.0,
    )
    measured = summarize_bands(probabilities, labels, edges)

    # 정상 5건에 상한 0.4이므로 2건까지 허용되고, 그 2건을 그대로 쓴다.
    assert measured["alert_false_positive_rate"] == pytest.approx(0.4)


def test_ceiling_of_one_allows_every_normal_into_the_alert_band() -> None:
    """상한이 1.0이면 경계를 끝까지 내려 경고 Recall이 최대"""
    probabilities, labels = build_separable_case()

    edges = select_standalone_bands(
        probabilities,
        labels,
        max_alert_false_positive_rate=1.0,
        min_coverage_recall=1.0,
    )
    measured = summarize_bands(probabilities, labels, edges)

    assert measured["alert_recall"] == 1.0
