"""Stacking 모델 학습, validation 임계값 선택 및 artifact 저장"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, recall_score

from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    SPLIT_MANIFEST_PATH,
    load_data,
    split_data,
)


SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent

STACKING_ARTIFACT_VERSION = "v2"

STACKING_ARTIFACT_DIRECTORY = (
    SMS_MODEL_DIRECTORY
    / "artifacts"
    / "stacking"
    / STACKING_ARTIFACT_VERSION
)

STACKING_MODEL_PATH = (
    STACKING_ARTIFACT_DIRECTORY / "model.joblib"
)

STACKING_METADATA_PATH = (
    STACKING_ARTIFACT_DIRECTORY / "metadata.json"
)

STACKING_REPORT_DIRECTORY = (
    SMS_MODEL_DIRECTORY
    / "reports"
    / "stacking_v2"
)

TARGET_RECALL = 0.95

def calculate_sha256(path: Path) -> str:
    """artifact 무결성 확인용 SHA-256 계산"""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

def select_validation_threshold(
    probabilities: np.ndarray,
    labels: pd.Series,
    *,
    target_recall: float = TARGET_RECALL,
) -> tuple[float, dict[str, float]]:
    """Recall 목표를 만족하는 후보 중 F2가 가장 높은 임계 값을 선택"""

    probability_array = np.asarray(probabilities, dtype=np.float64)

    if probability_array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    if len(probability_array) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    if len(probability_array) == 0:
        raise ValueError("validation data must not be empty")

    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")

    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1")

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be between 0 and 1")

    normalized_labels = labels.astype(str)
    supported_labels = {"normal", "phishing"}
    observed_labels = set(normalized_labels)

    if observed_labels != supported_labels:
        raise ValueError(
            "validation labels must contain normal and phishing"
        )

    binary_labels = (
        normalized_labels == "phishing"
    ).astype(int).to_numpy()

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                probability_array,
            ]
        )
    )

    best: tuple[float, float, float] | None = None

    for threshold in candidates:
        predictions = (probability_array >= threshold).astype(int)
        recall = recall_score(
            binary_labels,
            predictions,
            zero_division=0,
        )
        f2 = fbeta_score(
            binary_labels,
            predictions,
            beta=2,
            zero_division=0,
        )

        if recall < target_recall:
            continue

        candidate = (float(f2), float(recall), float(threshold))

        if best is None or candidate > best:
            best = candidate

    if best is None:
        # 목표 Recall을 만족하지 못하면 validation에서 Recall이
        # 최대가 되는 보수적인 최저 임계값을 사용
        threshold = float(np.min(candidates))
        predictions = (probability_array >= threshold).astype(int)

        return threshold, {
            "recall": float(
                recall_score(
                    binary_labels,
                    predictions,
                    zero_division=0,
                )
            ),
            "f2": float(
                fbeta_score(
                    binary_labels,
                    predictions,
                    beta=2,
                    zero_division=0,
                )
            ),
            "target_recall": target_recall,
            "target_recall_met": False,
        }

    f2, recall, threshold = best

    return threshold, {
        "recall": recall,
        "f2": f2,
        "target_recall": target_recall,
        "target_recall_met": True,
    }

def save_artifact(
        classifier: StackingPhishingClassifier,
    *,
    validation_metrics: dict[str, float],
    overwrite: bool,
) -> None:
    """모델과 비민감 metadata를 저장하고 재로드 검증"""

    if STACKING_MODEL_PATH.is_file() and not overwrite:
        raise FileExistsError(
            "stacking artifact already exists; "
            "use --overwrite-artifacts to replace it"
        )

    STACKING_ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "classifier": classifier,
    }

    joblib.dump(payload, STACKING_MODEL_PATH)

    loaded = joblib.load(STACKING_MODEL_PATH)

    if loaded.get("schema_version") != 1:
        raise RuntimeError("invalid stacking artifact schema")

    loaded_classifier = loaded.get("classifier")

    if not isinstance(
        loaded_classifier,
        StackingPhishingClassifier,
    ):
        raise RuntimeError("invalid stacking classifier artifact")

    metadata = {
        "schema_version": 2,
        "artifact_version": "v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": classifier.get_metadata(),
        "validation": validation_metrics,
        "dataset": {
            "total_csv_rows": 3002,
            "training_pool_rows": 885,
            "holdout_rows": 210,
            "dataset_fingerprint": (
                "46c1c9393d30f25ab03f0f7b6e85e5a"
                "8706f682f9bf2b68c872567b7eb5256f2"
            ),
        },
        "splits": {
            "train": 623,
            "validation": 126,
            "test": 136,
        },
        "training_policy": {
            "training_split": "train",
            "threshold_selection_split": "validation",
            "final_evaluation_split": "test",
            "holdout_used_during_training": False,
            "test_used_for_tuning": False,
            "random_state": 42,
        },
        "split_manifest": "sms_split_v2.csv",
        "model_sha256": calculate_sha256(
            STACKING_MODEL_PATH
        ),
        "library_versions": collect_library_versions(),
    }
    

    STACKING_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def train_stacking(*, overwrite_artifacts: bool) -> None:
    """Stacking v2를 학습하고 고정된 test split을 한 번 평가"""

    # committed v2 manifest가 없으면 실행 중단
    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: {SPLIT_MANIFEST_PATH}"
        )

    if SPLIT_MANIFEST_PATH.name != "sms_split_v2.csv":
        raise ValueError(
            "Stacking v2 must use sms_split_v2.csv"
        )

    # 3,002건 원본에서 학습 pool 885건과 holdout 210건 분리
    dataset, holdout = load_data(DATA_PATH)

    if len(dataset) != 885:
        raise ValueError(
            f"expected 885 training-pool rows, got {len(dataset)}"
        )

    if len(holdout) != 210:
        raise ValueError(
            f"expected 210 holdout rows, got {len(holdout)}"
        )

    # 기존 committed manifest만 적용
    splits = split_data(
        dataset,
        create_manifest=False,
    )

    expected_counts = {
        "train": 623,
        "validation": 126,
        "test": 136,
    }
    actual_counts = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }

    if actual_counts != expected_counts:
        raise ValueError(
            "unexpected split counts: "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    # train 623건만 사용해 모델 학습
    classifier = StackingPhishingClassifier(
        n_splits=5,
        random_state=42,
    )
    classifier.fit(splits.train)

    # validation 126건으로만 threshold 선정
    validation_probabilities, unavailable = (
        classifier.predict_probabilities(
            splits.validation
        )
    )

    if unavailable:
        raise RuntimeError(
            "base models unavailable during validation: "
            f"{unavailable}"
        )

    threshold, validation_metrics = (
        select_validation_threshold(
            validation_probabilities,
            splits.validation["label"],
            target_recall=TARGET_RECALL,
        )
    )

    # 이 시점부터 threshold는 변경하면 안 됨
    classifier.set_threshold(threshold)

    # 고정된 threshold로 test 136건을 단 한 번 평가
    test_probabilities, test_unavailable = (
        classifier.predict_probabilities(splits.test)
    )

    test_predictions = np.where(
        test_probabilities >= threshold,
        "phishing",
        "normal",
    )

    test_metrics = calculate_classification_metrics(
        splits.test["label"].to_numpy(),
        test_predictions,
    )

    # v2 전용 경로에 artifact 저장
    save_artifact(
        classifier,
        validation_metrics=validation_metrics,
        overwrite=overwrite_artifacts,
    )

    # 보고서 저장
    save_stacking_test_report(
        test_df=splits.test,
        probabilities=test_probabilities,
        predictions=test_predictions,
        metrics=test_metrics,
        threshold=threshold,
        unavailable_models=test_unavailable,
        output_directory=STACKING_REPORT_DIRECTORY,
    )

    print("[Stacking v2] training completed")
    print(f"  train={len(splits.train)}")
    print(f"  validation={len(splits.validation)}")
    print(f"  test={len(splits.test)}")
    print(f"  holdout={len(holdout)}")
    print(f"  threshold={threshold:.6f}")
    print(f"  test_accuracy={test_metrics.accuracy:.4f}")
    print(f"  test_recall={test_metrics.recall:.4f}")
    print(f"  test_f2={test_metrics.f2:.4f}")
    print(f"  artifact={STACKING_MODEL_PATH}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the SafeFam stacking phishing classifier."
    )
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
    )
    arguments = parser.parse_args()

    train_stacking(
        overwrite_artifacts=arguments.overwrite_artifacts
    )

if __name__ == "__main__":
    main()
