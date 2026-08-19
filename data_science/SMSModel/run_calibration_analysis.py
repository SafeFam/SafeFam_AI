"""확률 캘리브레이션 효과 측정"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from data_science.SMSModel.evaluation.standalone_bands import (
    measure_edge_transfer,
    measure_reliability,
)
from data_science.SMSModel.modeling.calibration import (
    CALIBRATION_METHODS,
    brier_score,
    expected_calibration_error,
    fit_probability_calibrator,
)
from data_science.SMSModel.run_stacking_training import (
    STACKING_MODEL_PATH,
)
from data_science.SMSModel.run_error_analysis import load_classifier
from data_science.SMSModel.run_standalone_analysis import (
    ADOPTION_CRITERIA,
    SELECTION_SPLIT,
    SPLIT_ORDER,
    collect_scored_splits,
    find_reference_edges,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
# artifact 버전은 학습 스크립트를 따른다. 여기서 따로 고정하지 않는다.
DEFAULT_MODEL_PATH = STACKING_MODEL_PATH
DEFAULT_OUTPUT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "calibration_analysis.json"
)


def measure_quality(probabilities, labels) -> dict[str, float]:
    """캘리브레이션 품질과 순위 보존 여부를 함께 측정"""
    truth = (labels == "phishing").astype(int)

    return {
        "expected_calibration_error": expected_calibration_error(
            probabilities, labels
        ),
        "brier_score": brier_score(probabilities, labels),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
    }


def apply_calibrator(calibrator, scored: dict[str, tuple]) -> dict[str, tuple]:
    """모든 split에 같은 캘리브레이터를 적용"""
    return {
        name: (calibrator.transform(probabilities), labels)
        for name, (probabilities, labels) in scored.items()
    }


def measure_variant(scored: dict[str, tuple]) -> dict[str, object]:
    """한 확률 집합에 대한 split별 품질과 경계 전이를 측정"""
    splits = {
        name: {
            "sample_count": int(len(probabilities)),
            "quality": measure_quality(probabilities, labels),
            "reliability": measure_reliability(probabilities, labels),
        }
        for name, (probabilities, labels) in scored.items()
    }

    reference_edges = find_reference_edges(*scored[SELECTION_SPLIT])

    return {
        "splits": splits,
        "reference_edges": (
            reference_edges.to_dict() if reference_edges else None
        ),
        "edge_transfer": (
            measure_edge_transfer(reference_edges, SELECTION_SPLIT, scored)
            if reference_edges
            else []
        ),
    }


def build_report(classifier) -> dict[str, object]:
    """보정 전과 방법별 보정 후를 한데 모음"""
    scored = collect_scored_splits(classifier)

    variants = {"uncalibrated": measure_variant(scored)}

    for method in CALIBRATION_METHODS:
        calibrator = fit_probability_calibrator(
            *scored[SELECTION_SPLIT],
            method=method,
        )
        variant = measure_variant(apply_calibrator(calibrator, scored))
        variant["calibrator"] = calibrator.to_dict()
        variants[method] = variant

    return {
        "calibration_split": SELECTION_SPLIT,
        "reference_alert_false_positive_target": (
            ADOPTION_CRITERIA.max_alert_false_positive_rate
        ),
        "reference_coverage_recall_target": (
            ADOPTION_CRITERIA.min_coverage_recall
        ),
        "variants": variants,
    }


def print_summary(report: dict[str, object]) -> None:
    """사람이 읽을 요약을 표준 출력에 남김"""
    print(f"[Calibration] 보정 split={report['calibration_split']}")

    for name, variant in report["variants"].items():
        print(f"\n  === {name} ===")
        for split in SPLIT_ORDER:
            quality = variant["splits"][split]["quality"]
            print(
                f"    {split:<14} "
                f"ECE {quality['expected_calibration_error']:.4f}"
                f" · Brier {quality['brier_score']:.4f}"
                f" · ROC-AUC {quality['roc_auc']:.4f}"
            )

        if not variant["edge_transfer"]:
            print("    [전이] 기준 경계를 달성할 수 없어 측정하지 않음")
            continue

        edges = variant["reference_edges"]
        print(
            f"    [전이] normal_max={edges['normal_max']:.4f}"
            f" phishing_min={edges['phishing_min']:.4f}"
        )
        for transfer in variant["edge_transfer"]:
            print(
                f"      {transfer['split']:<14} "
                f"경고FPR {transfer['alert_false_positive_rate']:.3f}"
                f" ({transfer['alert_false_positive_rate_drift']:+.3f})"
                f" · 포착 {transfer['coverage_recall']:.3f}"
                f" · 의심 {transfer['uncertain_share']:.3f}"
            )


def main() -> None:
    """CLI 인자를 읽어 캘리브레이션 측정 리포트를 JSON으로 저장"""
    parser = argparse.ArgumentParser(
        description="Measure the effect of probability calibration."
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

    print(f"[Calibration] {arguments.output}")
    print_summary(report)


if __name__ == "__main__":
    main()
