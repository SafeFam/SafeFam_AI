"""단독 운영 가능성 측정"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_science.SMSModel.evaluation.standalone_bands import (
    BandEdges,
    measure_edge_transfer,
    measure_reliability,
    summarize_bands,
    sweep_band_frontier,
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
DEFAULT_MODEL_PATH = (
    SMS_MODEL_DIRECTORY / "artifacts" / "stacking" / "v3" / "model.joblib"
)
DEFAULT_OUTPUT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "standalone_analysis.json"
)

SELECTION_SPLIT = "validation"

SPLIT_ORDER = ("validation", "test", "real_holdout")


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
        frame = score_frame(classifier, frames[name])
        scored[name] = (
            frame["probability"].to_numpy(),
            frame["label"].to_numpy(),
        )

    return scored


def find_reference_edges(
    frontier: list[dict[str, object]],
) -> BandEdges | None:
    """전이 확인의 기준으로 삼을 경계를 고름"""
    for entry in frontier:
        if (
            entry["target_alert_false_positive_rate"] == 0.01
            and entry["target_coverage_recall"] == 0.95
            and entry.get("feasible")
        ):
            measured = entry["measured"]
            return BandEdges(**measured["edges"])

    return None


def build_report(classifier) -> dict[str, object]:
    """split별 구간 측정과 경계 전이 결과를 한데 모음"""
    scored = collect_scored_splits(classifier)
    artifact_threshold = float(classifier.threshold)

    splits: list[dict[str, object]] = []
    for name in SPLIT_ORDER:
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

    selection_frontier = next(
        split["band_frontier"]
        for split in splits
        if split["split"] == SELECTION_SPLIT
    )
    reference_edges = find_reference_edges(selection_frontier)

    return {
        "artifact_threshold": artifact_threshold,
        "selection_split": SELECTION_SPLIT,
        "reference_edges": (
            reference_edges.to_dict() if reference_edges else None
        ),
        "edge_transfer": (
            measure_edge_transfer(reference_edges, SELECTION_SPLIT, scored)
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
