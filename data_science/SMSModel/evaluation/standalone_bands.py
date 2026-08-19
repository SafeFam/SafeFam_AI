"""단독 운영 관점의 2-임계값 구간 측정"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# 자동 경고를 낼 때 감수할 정상 오탐률 후보
ALERT_FALSE_POSITIVE_TARGETS = (0.005, 0.01, 0.02, 0.05)

# 최소 '의심' 이상으로는 걸러내야 하는 피싱 비율 후보
COVERAGE_RECALL_TARGETS = (0.90, 0.95, 0.98, 1.00)

# 신뢰도 곡선을 확인할 확률 구간
RELIABILITY_BIN_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)

CERTAIN_NORMAL = "certain_normal"
UNCERTAIN = "uncertain"
CERTAIN_PHISHING = "certain_phishing"

BAND_NAMES = (CERTAIN_NORMAL, UNCERTAIN, CERTAIN_PHISHING)


@dataclass(frozen=True)
class BandEdges:
    """3구간을 가르는 두 경계"""

    normal_max: float
    phishing_min: float

    def __post_init__(self) -> None:
        for name, value in (
            ("normal_max", self.normal_max),
            ("phishing_min", self.phishing_min),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.normal_max > self.phishing_min:
            raise ValueError("normal_max must not exceed phishing_min")

    def to_dict(self) -> dict[str, float]:
        """리포트에 기록할 dictionary로 변환"""
        return asdict(self)


def _validate_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """확률과 label 입력의 무결성을 검사하고 정규화"""
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=str)

    if probability_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("probabilities and labels must be one-dimensional")

    if len(probability_array) != len(label_array):
        raise ValueError("probabilities and labels must have the same length")

    if len(probability_array) == 0:
        raise ValueError("cannot measure bands from empty inputs")

    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")

    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1")

    allowed_labels = {"normal", "phishing"}
    observed_labels = set(label_array)

    if not observed_labels.issubset(allowed_labels):
        raise ValueError(
            f"unsupported labels: {observed_labels - allowed_labels}"
        )

    if observed_labels != allowed_labels:
        raise ValueError("labels must contain both normal and phishing")

    return probability_array, label_array


def assign_bands(
    probabilities,
    edges: BandEdges,
) -> np.ndarray:
    """확률을 3구간 이름으로 매핑"""
    probability_array = np.asarray(probabilities, dtype=np.float64)

    if probability_array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    return np.where(
        probability_array >= edges.phishing_min,
        CERTAIN_PHISHING,
        np.where(
            probability_array <= edges.normal_max,
            CERTAIN_NORMAL,
            UNCERTAIN,
        ),
    )


def summarize_bands(
    probabilities,
    labels,
    edges: BandEdges,
) -> dict[str, object]:
    """구간별 표본 분포와 단독 운영에 직결되는 두 오류율을 계산"""
    probability_array, label_array = _validate_inputs(probabilities, labels)
    bands = assign_bands(probability_array, edges)

    is_phishing = label_array == "phishing"
    phishing_total = int(is_phishing.sum())
    normal_total = int((~is_phishing).sum())

    by_band: dict[str, dict[str, float]] = {}
    for band in BAND_NAMES:
        in_band = bands == band
        phishing_in_band = int((in_band & is_phishing).sum())
        normal_in_band = int((in_band & ~is_phishing).sum())

        by_band[band] = {
            "sample_count": int(in_band.sum()),
            "phishing_count": phishing_in_band,
            "normal_count": normal_in_band,
            # 각 클래스가 이 구간으로 얼마나 흘러들어오는지
            "phishing_share": (
                phishing_in_band / phishing_total if phishing_total else 0.0
            ),
            "normal_share": (
                normal_in_band / normal_total if normal_total else 0.0
            ),
        }

    return {
        "edges": edges.to_dict(),
        "sample_count": len(probability_array),
        "phishing_count": phishing_total,
        "normal_count": normal_total,
        "by_band": by_band,
        "alert_false_positive_rate": by_band[CERTAIN_PHISHING]["normal_share"],
        "alert_recall": by_band[CERTAIN_PHISHING]["phishing_share"],
        "coverage_recall": 1.0 - by_band[CERTAIN_NORMAL]["phishing_share"],
        "missed_phishing_rate": by_band[CERTAIN_NORMAL]["phishing_share"],
        "uncertain_share": (
            by_band[UNCERTAIN]["sample_count"] / len(probability_array)
        ),
    }


def _lowest_alert_edge(
    probabilities: np.ndarray,
    is_phishing: np.ndarray,
    max_false_positive_rate: float,
) -> float | None:
    """정상 오탐률 상한을 지키는 가장 낮은 자동 경고 경계"""
    normal_probabilities = probabilities[~is_phishing]
    normal_total = len(normal_probabilities)

    if normal_total == 0:
        return None

    allowed_false_positives = int(np.floor(max_false_positive_rate * normal_total))
    descending = np.sort(normal_probabilities)[::-1]

    if allowed_false_positives == 0:
        edge = float(np.nextafter(descending[0], np.inf))
    else:
        edge = float(descending[allowed_false_positives - 1])

    return edge if edge <= 1.0 else None


def _highest_normal_edge(
    probabilities: np.ndarray,
    is_phishing: np.ndarray,
    min_coverage_recall: float,
) -> float | None:
    """놓치는 피싱 예산을 지키는 가장 높은 무알림 경계"""
    phishing_probabilities = probabilities[is_phishing]
    phishing_total = len(phishing_probabilities)

    if phishing_total == 0:
        return None

    allowed_misses = int(np.floor((1.0 - min_coverage_recall) * phishing_total))
    ascending = np.sort(phishing_probabilities)

    if allowed_misses == 0:
        edge = float(np.nextafter(ascending[0], -np.inf))
    else:
        edge = float(ascending[allowed_misses - 1])

    return edge if edge >= 0.0 else None


def sweep_band_frontier(
    probabilities,
    labels,
    *,
    alert_false_positive_targets=ALERT_FALSE_POSITIVE_TARGETS,
    coverage_recall_targets=COVERAGE_RECALL_TARGETS,
) -> list[dict[str, object]]:
    """두 목표를 교차시켜 도달 가능한 구간 경계와 그 결과를 나열"""
    probability_array, label_array = _validate_inputs(probabilities, labels)
    is_phishing = label_array == "phishing"

    frontier: list[dict[str, object]] = []

    for alert_target in alert_false_positive_targets:
        if not 0.0 <= alert_target <= 1.0:
            raise ValueError(
                "alert false positive targets must be between 0 and 1"
            )

        phishing_min = _lowest_alert_edge(
            probability_array,
            is_phishing,
            alert_target,
        )

        for coverage_target in coverage_recall_targets:
            if not 0.0 < coverage_target <= 1.0:
                raise ValueError(
                    "coverage recall targets must be greater than 0 "
                    "and at most 1"
                )

            normal_max_limit = _highest_normal_edge(
                probability_array,
                is_phishing,
                coverage_target,
            )

            entry: dict[str, object] = {
                "target_alert_false_positive_rate": float(alert_target),
                "target_coverage_recall": float(coverage_target),
            }

            if phishing_min is None:
                entry["feasible"] = False
                entry["reason"] = "ALERT_CEILING_UNREACHABLE"
                frontier.append(entry)
                continue

            if normal_max_limit is None:
                entry["feasible"] = False
                entry["reason"] = "COVERAGE_TARGET_UNREACHABLE"
                frontier.append(entry)
                continue

            edges = BandEdges(
                normal_max=max(0.0, min(normal_max_limit, phishing_min)),
                phishing_min=max(0.0, min(1.0, phishing_min)),
            )

            entry["feasible"] = True
            entry["measured"] = summarize_bands(
                probability_array,
                label_array,
                edges,
            )
            frontier.append(entry)

    return frontier


def select_reference_edges(
    frontier: list[dict[str, object]],
    *,
    alert_false_positive_target: float,
    coverage_recall_target: float,
) -> BandEdges | None:
    """스윕 결과에서 목표 조합 하나에 해당하는 경계를 꺼냄"""
    for entry in frontier:
        if (
            entry["target_alert_false_positive_rate"]
            == alert_false_positive_target
            and entry["target_coverage_recall"] == coverage_recall_target
            and entry.get("feasible")
        ):
            return BandEdges(**entry["measured"]["edges"])

    return None


def measure_reliability(
    probabilities,
    labels,
    *,
    bin_edges=RELIABILITY_BIN_EDGES,
) -> list[dict[str, float]]:
    """확률 구간마다 예측 확률과 실제 피싱 비율을 비교"""
    probability_array, label_array = _validate_inputs(probabilities, labels)
    edges = np.asarray(bin_edges, dtype=np.float64)

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must contain at least two boundaries")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")

    is_phishing = label_array == "phishing"
    bins: list[dict[str, float]] = []

    for index in range(len(edges) - 1):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == len(edges) - 2:
            in_bin = (probability_array >= lower) & (probability_array <= upper)
        else:
            in_bin = (probability_array >= lower) & (probability_array < upper)

        sample_count = int(in_bin.sum())
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "sample_count": sample_count,
                "mean_predicted_probability": (
                    float(probability_array[in_bin].mean())
                    if sample_count
                    else 0.0
                ),
                "observed_phishing_rate": (
                    float(is_phishing[in_bin].mean()) if sample_count else 0.0
                ),
            }
        )

    return bins


def measure_edge_transfer(
    edges: BandEdges,
    selection_split: str,
    evaluation_splits: dict[str, tuple],
) -> list[dict[str, object]]:
    """한 split에서 고른 경계를 다른 split에 그대로 적용했을 때의 차이"""
    if selection_split not in evaluation_splits:
        raise ValueError(
            f"selection split not measured: {selection_split}"
        )

    baseline = summarize_bands(*evaluation_splits[selection_split], edges)

    transfers: list[dict[str, object]] = []
    for name, (probabilities, labels) in evaluation_splits.items():
        measured = summarize_bands(probabilities, labels, edges)
        transfers.append(
            {
                "split": name,
                "is_selection_split": name == selection_split,
                "alert_false_positive_rate": measured[
                    "alert_false_positive_rate"
                ],
                "coverage_recall": measured["coverage_recall"],
                "uncertain_share": measured["uncertain_share"],
                "alert_false_positive_rate_drift": (
                    measured["alert_false_positive_rate"]
                    - baseline["alert_false_positive_rate"]
                ),
                "coverage_recall_drift": (
                    measured["coverage_recall"] - baseline["coverage_recall"]
                ),
            }
        )

    return transfers
