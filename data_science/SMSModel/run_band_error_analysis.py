"""밴드 경계에 걸린 오분류 표본을 모아 개선 실험의 기준선을 설계"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from data_science.SMSModel.evaluation.adoption import AdoptionCriteria
from data_science.SMSModel.evaluation.standalone_bands import (
    CERTAIN_PHISHING,
    UNCERTAIN,
    BandEdges,
    select_standalone_bands,
    summarize_bands,
)
from data_science.SMSModel.run_error_analysis import (
    load_classifier,
    score_frame,
)
from data_science.SMSModel.run_stacking_training import STACKING_MODEL_PATH
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    load_data,
    select_real_holdout,
    split_data,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = STACKING_MODEL_PATH
DEFAULT_OUTPUT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "band_error_baseline.json"
)

SELECTION_SPLIT = "validation"
JUDGING_SPLIT = "real_holdout"

REPORT_SCHEMA_VERSION = 1


def score_splits(classifier) -> dict[str, pd.DataFrame]:
    """경계 선정용 split과 판정용 split에 확률"""
    pool, holdout = load_data(DATA_PATH)
    splits = split_data(pool, create_manifest=False)

    return {
        SELECTION_SPLIT: score_frame(classifier, splits.validation),
        JUDGING_SPLIT: score_frame(classifier, select_real_holdout(holdout)),
    }


def select_edges(selection: pd.DataFrame) -> BandEdges:
    """채택 기준과 동일한 방식으로 경계를 선정"""
    criteria = AdoptionCriteria()

    return select_standalone_bands(
        selection["probability"].to_numpy(),
        selection["label"].to_numpy(),
        max_alert_false_positive_rate=(
            criteria.max_alert_false_positive_rate
        ),
        min_coverage_recall=criteria.min_coverage_recall,
    )


def describe_samples(frame: pd.DataFrame) -> list[dict[str, object]]:
    """확률이 높은 순으로 표본을 기록"""
    ordered = frame.sort_values("probability", ascending=False)

    return [
        {
            "text_fingerprint": str(row["text_fingerprint"]),
            "type": str(row["type"]),
            "probability": float(row["probability"]),
        }
        for _, row in ordered.iterrows()
    ]


def count_by_type(
    frame: pd.DataFrame,
    population: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """유형별로 몇 건 중 몇 건이 걸렸는지 세어 개선 우선순위"""
    totals = population["type"].value_counts()
    caught = frame["type"].value_counts()

    return {
        str(message_type): {
            "count": int(count),
            "total": int(totals.get(message_type, 0)),
        }
        for message_type, count in caught.items()
    }


def measure_gaps(
    judged: dict[str, object],
    criteria: AdoptionCriteria,
) -> dict[str, dict[str, float]]:
    """두 기준이 각각 몇 건 모자란지 건수로 환산"""
    phishing_total = int(judged["phishing_count"])
    normal_total = int(judged["normal_count"])

    covered = round(judged["coverage_recall"] * phishing_total)
    required_covered = math.ceil(
        criteria.min_coverage_recall * phishing_total
    )

    uncertain_normals = int(judged["by_band"][UNCERTAIN]["normal_count"])
    allowed_uncertain = math.floor(
        criteria.max_uncertain_normal_share * normal_total
    )

    return {
        "coverage_recall": {
            "measured": float(judged["coverage_recall"]),
            "required": criteria.min_coverage_recall,
            "covered_phishing": int(covered),
            "required_phishing": int(required_covered),
            "shortfall": max(0, int(required_covered - covered)),
        },
        "uncertain_normal_share": {
            "measured": float(judged["by_band"][UNCERTAIN]["normal_share"]),
            "required": criteria.max_uncertain_normal_share,
            "uncertain_normals": uncertain_normals,
            "allowed_normals": int(allowed_uncertain),
            "excess": max(0, int(uncertain_normals - allowed_uncertain)),
        },
    }


def build_report(classifier) -> dict[str, object]:
    """경계 근처 오분류를 모아 기준선 리포트 생성"""
    criteria = AdoptionCriteria()
    scored = score_splits(classifier)
    edges = select_edges(scored[SELECTION_SPLIT])

    judging = scored[JUDGING_SPLIT]
    judged = summarize_bands(
        judging["probability"].to_numpy(),
        judging["label"].to_numpy(),
        edges,
    )

    is_phishing = judging["label"].eq("phishing")
    normals = judging[~is_phishing]

    missed = judging[is_phishing & judging["probability"].lt(edges.normal_max)]
    uncertain_normals = normals[
        normals["probability"].ge(edges.normal_max)
        & normals["probability"].lt(edges.phishing_min)
    ]

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "selection_split": SELECTION_SPLIT,
        "judging_split": JUDGING_SPLIT,
        "edges": edges.to_dict(),
        "judging_set": {
            "sample_count": int(len(judging)),
            "normal_count": int(judged["normal_count"]),
            "phishing_count": int(judged["phishing_count"]),
        },
        "gaps": measure_gaps(judged, criteria),
        "missed_phishing": {
            "count": int(len(missed)),
            "by_type": count_by_type(missed, judging[is_phishing]),
            "samples": describe_samples(missed),
        },
        "uncertain_normals": {
            "count": int(len(uncertain_normals)),
            "by_type": count_by_type(uncertain_normals, normals),
            "samples": describe_samples(uncertain_normals),
        },
        "alert_false_positive_rate": float(
            judged["by_band"][CERTAIN_PHISHING]["normal_share"]
        ),
    }


def print_summary(report: dict[str, object]) -> None:
    """개선 방향을 잡을 수 있을 만큼만 사람이 읽을 형태로 출력"""
    edges = report["edges"]
    print(
        f"[Band] normal_max={edges['normal_max']:.4f}"
        f" phishing_min={edges['phishing_min']:.4f}"
    )

    gaps = report["gaps"]
    coverage = gaps["coverage_recall"]
    uncertain = gaps["uncertain_normal_share"]
    print(
        f"  놓친 피싱 {report['missed_phishing']['count']}건"
        f" | recall {coverage['measured']:.4f},"
        f" 경계 위로 {coverage['shortfall']}건 더 올려야 한다"
    )
    print(
        f"  의심 정상 {report['uncertain_normals']['count']}건"
        f" | share {uncertain['measured']:.4f},"
        f" 경계 아래로 {uncertain['excess']}건 더 내려야 한다"
    )
    print("  두 요구는 같은 normal_max를 반대로 당긴다")

    for name in ("missed_phishing", "uncertain_normals"):
        print(f"\n  [{name}] 유형별")
        by_type = report[name]["by_type"]
        for message_type, counts in sorted(
            by_type.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        ):
            print(
                f"    {counts['count']:>3}/{counts['total']:<3}"
                f" {message_type}"
            )


def print_texts(scored: pd.DataFrame, report: dict[str, object]) -> None:
    """원문을 로컬에서만 출력. 리포트에는 남기지 않는다"""
    judging = scored.set_index("text_fingerprint")

    for name in ("missed_phishing", "uncertain_normals"):
        print(f"\n===== {name} =====")
        for sample in report[name]["samples"]:
            row = judging.loc[sample["text_fingerprint"]]
            print(f"\n--- {sample['type']} p={sample['probability']:.4f}")
            print(row["text"])


def main() -> None:
    """CLI 인자를 읽어 경계 오분류 기준선을 JSON으로 저장"""
    parser = argparse.ArgumentParser(
        description="Capture misclassifications around the standalone bands."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="원문을 콘솔에 출력한다. 리포트에는 저장하지 않는다.",
    )
    arguments = parser.parse_args()

    classifier = load_classifier(arguments.model_path)
    report = build_report(classifier)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[Band] {arguments.output}")
    print_summary(report)

    if arguments.show_text:
        print_texts(score_splits(classifier)[JUDGING_SPLIT], report)


if __name__ == "__main__":
    main()
