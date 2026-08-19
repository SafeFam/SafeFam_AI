"""오탐·미탐 원인 분석"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from data_science.SMSModel.run_stacking_training import (
    STACKING_MODEL_PATH,
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
DEFAULT_OUTPUT_PATH = SMS_MODEL_DIRECTORY / "reports" / "error_analysis.json"

# 오탐률 곡선을 확인할 지점
FPR_TARGETS = (0.02, 0.05, 0.10, 0.15, 0.20)

# 원인 파악용으로 들여다볼 표본 수
TOP_SAMPLE_COUNT = 15


def load_classifier(model_path: Path):
    """artifact에서 분류기 꺼냄"""
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "classifier" in payload:
        return payload["classifier"]
    return payload


def score_frame(classifier, frame: pd.DataFrame) -> pd.DataFrame:
    """확률을 붙인 사본을 반환"""
    probabilities, unavailable = classifier.predict_probabilities(frame)
    if unavailable:
        raise RuntimeError(f"base models unavailable: {unavailable}")

    scored = frame.copy()
    scored["probability"] = np.asarray(probabilities, dtype=float)
    return scored


def summarize_operating_points(scored: pd.DataFrame) -> list[dict[str, float]]:
    """목표 오탐률마다 도달 가능한 최대 Recall을 계산"""
    truth = (scored["label"].to_numpy() == "phishing").astype(int)
    probability = scored["probability"].to_numpy()

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        truth, probability
    )

    points = []
    for target in FPR_TARGETS:
        feasible = np.where(false_positive_rate <= target)[0]
        if feasible.size == 0:
            continue
        index = int(feasible.max())
        points.append(
            {
                "target_fpr": target,
                "max_recall": float(true_positive_rate[index]),
                "threshold": float(thresholds[index]),
                "missed_phishing": int(
                    round((1 - true_positive_rate[index]) * truth.sum())
                ),
            }
        )
    return points


def summarize_false_positives(
    scored: pd.DataFrame,
    threshold: float,
) -> dict[str, object]:
    """임계값을 넘은 정상 문자를 유형별로 집계"""
    normals = scored[scored["label"] == "normal"]
    flagged_total = int((normals["probability"] >= threshold).sum())

    by_type: dict[str, dict[str, float]] = {}
    for type_name, group in normals.groupby("type"):
        flagged = int((group["probability"] >= threshold).sum())
        by_type[str(type_name)] = {
            "sample_count": int(len(group)),
            "false_positive": flagged,
            "false_positive_rate": flagged / len(group),
        }

    return {
        "threshold": float(threshold),
        "normal_count": int(len(normals)),
        "false_positive_count": flagged_total,
        "false_positive_rate": (
            flagged_total / len(normals) if len(normals) else 0.0
        ),
        "by_type": dict(
            sorted(
                by_type.items(),
                key=lambda item: -item[1]["false_positive_rate"],
            )
        ),
    }


def list_extreme_samples(scored: pd.DataFrame) -> dict[str, list[dict]]:
    """임계값을 밀어올리는 정상과, 놓치기 쉬운 피싱 추출"""

    def to_records(frame: pd.DataFrame) -> list[dict]:
        """원문 없이 지문·유형·확률만 남긴 표본 레코드로 변환"""
        return [
            {
                "text_fingerprint": str(row.get("text_fingerprint", "")),
                "type": str(row["type"]),
                "probability": float(row["probability"]),
            }
            for _, row in frame.iterrows()
        ]

    normals = scored[scored["label"] == "normal"]
    phishing = scored[scored["label"] == "phishing"]

    return {
        # 확률이 높은 정상 = 임계값을 위로 미는 표본
        "highest_scoring_normals": to_records(
            normals.nlargest(TOP_SAMPLE_COUNT, "probability")
        ),
        # 확률이 낮은 피싱 = 임계값을 올리면 놓치는 표본
        "lowest_scoring_phishing": to_records(
            phishing.nsmallest(TOP_SAMPLE_COUNT, "probability")
        ),
    }


def analyze(classifier, name: str, frame: pd.DataFrame) -> dict[str, object]:
    """split 하나에 대한 분석 결과 생성"""
    scored = score_frame(classifier, frame)
    truth = (scored["label"].to_numpy() == "phishing").astype(int)
    probability = scored["probability"].to_numpy()

    return {
        "split": name,
        "sample_count": int(len(scored)),
        "phishing_count": int(truth.sum()),
        "normal_count": int(len(truth) - truth.sum()),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "operating_points": summarize_operating_points(scored),
        "false_positives_at_artifact_threshold": summarize_false_positives(
            scored, classifier.threshold
        ),
        "extreme_samples": list_extreme_samples(scored),
    }


def build_report(classifier) -> dict[str, object]:
    """validation·test·real_holdout 분석을 한데 모으기"""
    pool, holdout = load_data(DATA_PATH)
    splits = split_data(pool, create_manifest=False)

    return {
        "artifact_threshold": float(classifier.threshold),
        "structural_feature_names": list(
            classifier.get_metadata()["meta_feature_names"]
        ),
        "splits": [
            analyze(classifier, "validation", splits.validation),
            analyze(classifier, "test", splits.test),
            analyze(classifier, "real_holdout", select_real_holdout(holdout)),
        ],
    }


def main() -> None:
    """CLI 인자를 읽어 오탐·미탐 분석 리포트를 JSON으로 저장"""

    parser = argparse.ArgumentParser(
        description="Analyze false positives and missed phishing."
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

    print(f"[Error analysis] {arguments.output}")
    print(f"[Error analysis] threshold={report['artifact_threshold']:.6f}")

    for split in report["splits"]:
        print(
            f"  {split['split']:<14} "
            f"n={split['sample_count']:<4} "
            f"ROC-AUC {split['roc_auc']:.4f}"
        )
        for point in split["operating_points"]:
            if point["target_fpr"] == 0.05:
                print(
                    f"    FPR 5% → Recall {point['max_recall']:.4f} "
                    f"(피싱 {point['missed_phishing']}건 놓침)"
                )
        false_positives = split["false_positives_at_artifact_threshold"]
        worst = list(false_positives["by_type"].items())[:3]
        for type_name, stats in worst:
            if stats["false_positive"]:
                print(
                    f"    FP {stats['false_positive']}/{stats['sample_count']}"
                    f"  {type_name}"
                )


if __name__ == "__main__":
    main()
