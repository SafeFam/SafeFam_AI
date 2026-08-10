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
STACKING_ARTIFACT_DIRECTORY = (
    SMS_MODEL_DIRECTORY / "artifacts" / "stacking"
)
STACKING_MODEL_PATH = STACKING_ARTIFACT_DIRECTORY / "model.joblib"
STACKING_METADATA_PATH = STACKING_ARTIFACT_DIRECTORY / "metadata.json"

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

    binary_labels = (
        labels.astype(str) == "phishing"
    ).astype(int).to_numpy()

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                probabilities,
            ]
        )
    )

    best: tuple[float, float, float] | None = None

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
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
        predictions = (probabilities >= threshold).astype(int)

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

    if STACKING_ARTIFACT_DIRECTORY.exists() and not overwrite:
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

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": classifier.get_metadata(),
        "validation": validation_metrics,
        "dataset_path": DATA_PATH.as_posix(),
        "split_manifest": SPLIT_MANIFEST_PATH.name,
        "model_sha256": calculate_sha256(STACKING_MODEL_PATH),
        # 원문 데이터, API Key 및 환경변수는 기록하지 않습니다.
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

    # 저장 직후 로드하여 손상되거나 불완전한 artifact를 방지
    loaded = joblib.load(STACKING_METADATA_PATH)

    if loaded.get("schema_version") != 1:
        raise RuntimeError("invalid stacking artifact schema")

    loaded_classifier = loaded.get("classifier")

    if not isinstance(
        loaded_classifier,
        StackingPhishingClassifier,
    ):
        raise RuntimeError("invalid stacking classifier artifact")

def train_stacking(*, overwrite_artifacts: bool) -> None:
    """train으로 학습하고 validation으로 임계값을 선택"""

    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: {SPLIT_MANIFEST_PATH}"
        )

    dataset, _unused_holdout = load_data(DATA_PATH)
    splits = split_data(dataset, create_manifest=False)

    classifier = StackingPhishingClassifier(
        n_splits=5,
        random_state=42,
    )
    classifier.fit(splits.train)

    validation_probabilities, unavailable = (
        classifier.predict_probabilities(splits.validation)
    )

    if unavailable:
        raise RuntimeError(
            "all base models must be available during threshold selection: "
            f"{unavailable}"
        )

    threshold, metrics = select_validation_threshold(
        validation_probabilities,
        splits.validation["label"],
    )
    classifier.set_threshold(threshold)

    save_artifact(
        classifier,
        validation_metrics=metrics,
        overwrite=overwrite_artifacts,
    )

    print("[Stacking] training completed")
    print(f"  threshold={threshold:.6f}")
    print(f"  validation_recall={metrics['recall']:.4f}")
    print(f"  validation_f2={metrics['f2']:.4f}")
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