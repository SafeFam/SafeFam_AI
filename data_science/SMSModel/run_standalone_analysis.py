"""단독 운영 가능성 측정"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_science.SMSModel.evaluation.adoption import AdoptionCriteria
from data_science.SMSModel.evaluation.standalone_bands import (
    BandEdges,
    BandEdgesUnreachableError,
    measure_edge_transfer,
    measure_reliability,
    select_standalone_bands,
    summarize_bands,
    sweep_band_frontier,
)
from data_science.SMSModel.evaluation.threshold import (
    ThresholdInfeasibleError,
    select_probability_threshold,
)
from data_science.SMSModel.run_stacking_training import (
    MAX_NORMAL_FALSE_POSITIVE_RATE,
    STACKING_MODEL_PATH,
    TARGET_RECALL,
)
from data_science.SMSModel.run_error_analysis import (
    load_classifier,
    score_frame,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    load_data,
    select_real_holdout,
    split_data,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
# artifact 버전은 학습 스크립트를 따른다. 여기서 따로 고정하지 않는다.
DEFAULT_MODEL_PATH = STACKING_MODEL_PATH
DEFAULT_OUTPUT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "standalone_analysis.json"
)

# train pool OOF(정정 규모 ~5배)로 threshold/구간을 선정한다. validation
# 수십~백여 건만으로 뽑으면 "오탐 0건" 같은 극값 통계라 표본이 몇 건만
# 바뀌어도 경계가 크게 흔들린다(#102 §5, 부트스트랩으로 phishing_min
# 5~95% 구간이 0.39~0.72까지 벌어짐을 확인). classifier.oof_probabilities_가
# 없는 구버전 artifact는 validation으로 자동 폴백한다.
SELECTION_SPLIT = "train_oof"
_FALLBACK_SELECTION_SPLIT = "validation"

SPLIT_ORDER = ("train_oof", "validation", "test", "real_holdout")

ADOPTION_CRITERIA = AdoptionCriteria()


def collect_scored_splits(classifier) -> dict[str, tuple]:
    """split마다 확률과 label을 붙여 반환"""
    pool, holdout = load_data(DATA_PATH)
    splits = split_data(pool, create_manifest=False)

    frames = {
        "validation": splits.validation,
        "test": splits.test,
        "real_holdout": select_real_holdout(holdout),
    }

    scored: dict[str, tuple] = {}
    for name in SPLIT_ORDER:
        if name == "train_oof":
            oof_probabilities = getattr(classifier, "oof_probabilities_", None)
            oof_labels = getattr(classifier, "oof_labels_", None)
            if oof_probabilities is None or oof_labels is None:
                # 구버전 artifact 호환 - OOF가 없으면 이 항목은 건너뛴다.
                continue
            scored[name] = (oof_probabilities, oof_labels)
            continue

        frame = score_frame(classifier, frames[name])
        scored[name] = (
            frame["probability"].to_numpy(),
            frame["label"].to_numpy(),
        )

    return scored


def resolve_selection_split(scored: dict[str, tuple]) -> str:
    """OOF가 없는 구버전 artifact는 validation으로 자동 폴백"""
    if SELECTION_SPLIT in scored:
        return SELECTION_SPLIT
    return _FALLBACK_SELECTION_SPLIT


def find_reference_edges(probabilities, labels) -> BandEdges | None:
    """전이 확인의 기준으로 삼을 경계를 채택 기준대로 선정"""
    try:
        return select_standalone_bands(
            probabilities,
            labels,
            max_alert_false_positive_rate=(
                ADOPTION_CRITERIA.max_alert_false_positive_rate
            ),
            min_coverage_recall=ADOPTION_CRITERIA.min_coverage_recall,
        )
    except BandEdgesUnreachableError:
        return None


def compare_threshold_policies(
    probabilities,
    labels,
) -> dict[str, object]:
    """상한 도입 전후로 어떤 임계값이 선택되는지 나란히 기록"""
    before = select_probability_threshold(
        probabilities,
        labels,
        target_recall=TARGET_RECALL,
    )

    entry: dict[str, object] = {
        "target_recall": TARGET_RECALL,
        "max_false_positive_rate": MAX_NORMAL_FALSE_POSITIVE_RATE,
        "before": before.to_validation_metrics() | {"threshold": before.threshold},
    }

    try:
        after = select_probability_threshold(
            probabilities,
            labels,
            target_recall=TARGET_RECALL,
            max_false_positive_rate=MAX_NORMAL_FALSE_POSITIVE_RATE,
        )
    except ThresholdInfeasibleError as error:
        entry["after"] = {
            "feasible": False,
            "best_recall_within_ceiling": error.best_recall_within_ceiling,
            "lowest_false_positive_rate_at_target_recall": (
                error.lowest_false_positive_rate_at_target_recall
            ),
            "measurable_false_positive_rate": (
                error.measurable_false_positive_rate
            ),
            "reason": str(error),
        }
    else:
        entry["after"] = after.to_validation_metrics() | {
            "feasible": True,
            "threshold": after.threshold,
        }

    return entry


def build_report(classifier) -> dict[str, object]:
    """split별 구간 측정과 경계 전이 결과를 한데 모음"""
    scored = collect_scored_splits(classifier)
    artifact_threshold = float(classifier.threshold)
    selection_split = resolve_selection_split(scored)

    splits: list[dict[str, object]] = []
    for name in SPLIT_ORDER:
        if name not in scored:
            continue
        probabilities, labels = scored[name]
        splits.append(
            {
                "split": name,
                "sample_count": int(len(probabilities)),
                "single_threshold_baseline": summarize_bands(
                    probabilities,
                    labels,
                    BandEdges(
                        normal_max=float(
                            np.nextafter(artifact_threshold, -np.inf)
                        ),
                        phishing_min=artifact_threshold,
                    ),
                ),
                "band_frontier": sweep_band_frontier(probabilities, labels),
                "reliability": measure_reliability(probabilities, labels),
            }
        )

    reference_edges = find_reference_edges(*scored[selection_split])

    return {
        "artifact_threshold": artifact_threshold,
        "selection_split": selection_split,
        "threshold_policy": compare_threshold_policies(*scored[selection_split]),
        "reference_edges": (
            reference_edges.to_dict() if reference_edges else None
        ),
        "edge_transfer": (
            measure_edge_transfer(reference_edges, selection_split, scored)
            if reference_edges
            else []
        ),
        "splits": splits,
    }


def print_summary(report: dict[str, object]) -> None:
    """사람이 읽을 요약을 표준 출력에 남김"""
    print(f"[Standalone] threshold={report['artifact_threshold']:.6f}")

    for split in report["splits"]:
        baseline = split["single_threshold_baseline"]
        print(
            f"\n  {split['split']:<14} n={split['sample_count']}"
            f"  단일 임계값 recall {baseline['alert_recall']:.3f}"
            f" · FPR {baseline['alert_false_positive_rate']:.3f}"
        )

        for entry in split["band_frontier"]:
            alert_target = entry["target_alert_false_positive_rate"]
            coverage_target = entry["target_coverage_recall"]
            label = (
                f"    경고FPR≤{alert_target:.3f} "
                f"· 포착≥{coverage_target:.2f}"
            )

            if not entry["feasible"]:
                print(f"{label}  →  불가 ({entry['reason']})")
                continue

            measured = entry["measured"]
            print(
                f"{label}  →  자동경고 recall {measured['alert_recall']:.3f}"
                f" · 의심 {measured['uncertain_share']:.3f}"
                f" · 놓침 {measured['missed_phishing_rate']:.3f}"
            )

    policy = report["threshold_policy"]
    before = policy["before"]
    after = policy["after"]
    print(
        f"\n  [임계값 정책] Recall>={policy['target_recall']}"
        f" · 정상 FPR<={policy['max_false_positive_rate']}"
    )
    print(
        f"    상한 없음  threshold {before['threshold']:.4f}"
        f" · recall {before['recall']:.3f}"
        f" · FPR {before['false_positive_rate']:.4f}"
    )
    if after["feasible"]:
        print(
            f"    상한 적용  threshold {after['threshold']:.4f}"
            f" · recall {after['recall']:.3f}"
            f" · FPR {after['false_positive_rate']:.4f}"
        )
    else:
        print(
            f"    상한 적용  불가 — 상한 안에서 Recall 최대 "
            f"{after['best_recall_within_ceiling']:.3f}, "
            f"목표 Recall은 FPR "
            f"{after['lowest_false_positive_rate_at_target_recall']:.4f} 요구"
        )

    if not report["edge_transfer"]:
        print("\n  [전이] 기준 경계를 달성할 수 없어 측정하지 않음")
        return

    edges = report["reference_edges"]
    print(
        f"\n  [전이] {report['selection_split']}에서 고른 경계 "
        f"normal_max={edges['normal_max']:.4f} "
        f"phishing_min={edges['phishing_min']:.4f}"
    )
    for transfer in report["edge_transfer"]:
        print(
            f"    {transfer['split']:<14} "
            f"경고FPR {transfer['alert_false_positive_rate']:.3f} "
            f"({transfer['alert_false_positive_rate_drift']:+.3f}) "
            f"· 포착 {transfer['coverage_recall']:.3f} "
            f"({transfer['coverage_recall_drift']:+.3f})"
        )


def main() -> None:
    """CLI 인자를 읽어 단독 운영 측정 리포트를 JSON으로 저장"""
    parser = argparse.ArgumentParser(
        description="Measure standalone operating bands for a stacking artifact."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    classifier = load_classifier(arguments.model_path)
    report = build_report(classifier)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[Standalone] {arguments.output}")
    print_summary(report)


if __name__ == "__main__":
    main()
