"""Stacking 모델 학습, validation 임계값 선택 및 artifact 저장"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_science.SMSModel.evaluation.metrics import (
    calculate_classification_metrics,
)
from data_science.SMSModel.evaluation.threshold import (
    select_probability_threshold,
)
from data_science.SMSModel.evaluation.stacking_reporting import (
    predict_probabilities_with_latency,
    save_stacking_test_report,
)
from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    DATASET_SPLIT_JSON_REPORT_PATH,
    SPLIT_MANIFEST_PATH,
    load_data,
    split_data,
)


SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent

STACKING_ARTIFACT_VERSION = "v4"

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
    / f"stacking_{STACKING_ARTIFACT_VERSION}"
)

EXPERIMENT_SUFFIX = "-experiment"

TARGET_RECALL = 0.95

MAX_NORMAL_FALSE_POSITIVE_RATE = 0.01
EXPECTED_DATASET_FINGERPRINT = (
    "29d0aa8b4faaf4a57490f3fe7d1b102"
    "b3689551a924089fd776791eb54912793"
)

EXPECTED_TOTAL_CSV_ROWS = 3242
EXPECTED_TRAINING_POOL_ROWS = 968
EXPECTED_HOLDOUT_ROWS = 280
EXPECTED_SPLIT_COUNTS = {
    "train": 683,
    "validation": 146,
    "test": 139,
}


def collect_library_versions() -> dict[str, str]:
    """재현성 확인에 필요한 실행 환경과 라이브러리 버전을 반환"""
    packages = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
    }
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }

    for key, package_name in packages.items():
        try:
            versions[key] = version(package_name)
        except PackageNotFoundError:
            versions[key] = "not-installed"

    return versions

def calculate_sha256(path: Path) -> str:
    """artifact 무결성 확인용 SHA-256 계산"""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_json_sha256(value: dict[str, object]) -> str:
    """정렬된 JSON 설정의 결정적인 SHA-256을 반환"""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_dataset_fingerprint() -> str:
    """확정한 데이터 fingerprint와 split 검증 상태를 확인"""
    if not DATASET_SPLIT_JSON_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "dataset split summary is required: "
            f"{DATASET_SPLIT_JSON_REPORT_PATH}"
        )
    report = json.loads(
        DATASET_SPLIT_JSON_REPORT_PATH.read_text(encoding="utf-8")
    )
    if report.get("validation", {}).get("passed") is not True:
        raise ValueError("dataset split validation did not pass")
    actual = report.get("dataset_fingerprint")
    if actual != EXPECTED_DATASET_FINGERPRINT:
        raise ValueError(
            "dataset fingerprint does not match #77: "
            f"expected={EXPECTED_DATASET_FINGERPRINT}, actual={actual}"
        )
    return str(actual)

def resolve_artifact_paths(
    max_false_positive_rate: float,
) -> tuple[Path, Path]:
    """정책 상한을 지켰는지에 따라 artifact와 리포트 경로 선택""
    if max_false_positive_rate == MAX_NORMAL_FALSE_POSITIVE_RATE:
        return STACKING_ARTIFACT_DIRECTORY, STACKING_REPORT_DIRECTORY

    return (
        STACKING_ARTIFACT_DIRECTORY.with_name(
            STACKING_ARTIFACT_DIRECTORY.name + EXPERIMENT_SUFFIX
        ),
        STACKING_REPORT_DIRECTORY.with_name(
            STACKING_REPORT_DIRECTORY.name + EXPERIMENT_SUFFIX
        ),
    )


def save_artifact(
        classifier: StackingPhishingClassifier,
    *,
    validation_metrics: dict[str, float],
    overwrite: bool,
    dataset_counts: dict[str, int],
    split_counts: dict[str, int],
    artifact_directory: Path | None = None,
    verification_df: pd.DataFrame | None = None,
    expected_probabilities: np.ndarray | None = None,
) -> None:
    """모델과 비민감 metadata를 저장하고 재로드 검증"""

    if artifact_directory is None:
        artifact_directory = STACKING_ARTIFACT_DIRECTORY

    model_path = artifact_directory / "model.joblib"
    metadata_path = artifact_directory / "metadata.json"

    missing_dataset_keys = {
        "total_csv_rows",
        "training_pool_rows",
        "holdout_rows",
    } - set(dataset_counts)
    if missing_dataset_keys:
        raise ValueError(
            f"dataset_counts is missing keys: {sorted(missing_dataset_keys)}"
        )

    missing_split_keys = set(EXPECTED_SPLIT_COUNTS) - set(split_counts)
    if missing_split_keys:
        raise ValueError(
            f"split_counts is missing keys: {sorted(missing_split_keys)}"
        )

    if (
        model_path.exists()
        or metadata_path.exists()
    ) and not overwrite:
        raise FileExistsError(
            "stacking artifact already exists; "
            "use --overwrite-artifacts to replace it"
        )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "classifier": classifier,
    }

    joblib.dump(payload, model_path)

    loaded = joblib.load(model_path)

    if loaded.get("schema_version") != 1:
        raise RuntimeError("invalid stacking artifact schema")

    loaded_classifier = loaded.get("classifier")

    if not isinstance(
        loaded_classifier,
        StackingPhishingClassifier,
    ):
        raise RuntimeError("invalid stacking classifier artifact")

    if verification_df is not None:
        if expected_probabilities is None:
            raise ValueError(
                "expected_probabilities are required with verification_df"
            )
        reloaded_probabilities, unavailable = (
            loaded_classifier.predict_probabilities(verification_df)
        )
        if unavailable:
            raise RuntimeError(
                "reloaded artifact has unavailable base models: "
                f"{unavailable}"
            )
        reloaded_array = np.asarray(reloaded_probabilities, dtype=float)
        expected_array = np.asarray(expected_probabilities, dtype=float)
        if reloaded_array.shape != expected_array.shape:
            raise RuntimeError(
                "reloaded artifact returned a different probability shape: "
                f"expected={expected_array.shape}, "
                f"actual={reloaded_array.shape}"
            )
        if not np.allclose(
            reloaded_array,
            expected_array,
            rtol=0.0,
            atol=1e-12,
        ):
            max_absolute_difference = float(
                np.max(np.abs(reloaded_array - expected_array))
            )
            raise RuntimeError(
                "reloaded artifact probabilities differ from training output: "
                f"max_abs_diff={max_absolute_difference}"
            )

    model_configuration = classifier.get_metadata()
    metadata = {
        "schema_version": 2,
        "artifact_version": STACKING_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_configuration,
        "validation": validation_metrics,
        "dataset": {
            "total_csv_rows": dataset_counts["total_csv_rows"],
            "training_pool_rows": dataset_counts["training_pool_rows"],
            "holdout_rows": dataset_counts["holdout_rows"],
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        },
        "splits": dict(split_counts),
        "training_policy": {
            "training_split": "train",
            "threshold_selection_split": "validation",
            "final_evaluation_split": "test",
            "holdout_used_during_training": False,
            "test_used_for_tuning": False,
            "random_state": 42,
        },
        "split_manifest": SPLIT_MANIFEST_PATH.name,
        "split_manifest_sha256": calculate_sha256(SPLIT_MANIFEST_PATH),
        "model_sha256": calculate_sha256(model_path),
        "model_configuration_sha256": calculate_json_sha256(
            model_configuration
        ),
        "library_versions": collect_library_versions(),
        "dataset_path": DATA_PATH.relative_to(
            SMS_MODEL_DIRECTORY.parent
        ).as_posix(),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

def train_stacking(
    *,
    overwrite_artifacts: bool,
    max_false_positive_rate: float = MAX_NORMAL_FALSE_POSITIVE_RATE,
) -> None:
    """Stacking artifact를 학습하고 고정된 test split을 한 번 평가"""

    # committed v4 manifest가 없으면 실행 중단
    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: {SPLIT_MANIFEST_PATH}"
        )

    if SPLIT_MANIFEST_PATH.name != "sms_split_v4.csv":
        raise ValueError(
            "Stacking must use sms_split_v4.csv"
        )

    # 3,002건 원본에서 학습 pool 885건과 holdout 210건 분리
    total_csv_rows = len(pd.read_csv(DATA_PATH))
    if total_csv_rows != EXPECTED_TOTAL_CSV_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_CSV_ROWS} source rows, "
            f"got {total_csv_rows}"
        )

    validate_dataset_fingerprint()

    dataset, holdout = load_data(DATA_PATH)

    if len(dataset) != EXPECTED_TRAINING_POOL_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TRAINING_POOL_ROWS} training-pool rows, "
            f"got {len(dataset)}"
        )

    if len(holdout) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError(
            f"expected {EXPECTED_HOLDOUT_ROWS} holdout rows, "
            f"got {len(holdout)}"
        )

    # 기존 committed manifest만 적용
    splits = split_data(
        dataset,
        create_manifest=False,
    )

    validate_dataset_fingerprint()

    dataset_counts = {
        "total_csv_rows": total_csv_rows,
        "training_pool_rows": len(dataset),
        "holdout_rows": len(holdout),
    }

    actual_counts = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }

    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "unexpected split counts: "
            f"expected={EXPECTED_SPLIT_COUNTS}, actual={actual_counts}"
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

    threshold_selection = select_probability_threshold(
        validation_probabilities,
        splits.validation["label"],
        target_recall=TARGET_RECALL,
        max_false_positive_rate=max_false_positive_rate,
    )
    threshold = threshold_selection.threshold
    validation_metrics = threshold_selection.to_validation_metrics()

    classifier.set_threshold(threshold)

    # 고정된 threshold로 test 136건을 단 한 번 평가
    test_probabilities, test_unavailable, test_latencies_ms = (
        predict_probabilities_with_latency(
            classifier,
            splits.test,
        )
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

    # 정책 상한을 지킨 실행만 정규 경로에 저장된다
    artifact_directory, report_directory = resolve_artifact_paths(
        max_false_positive_rate
    )

    save_artifact(
        classifier,
        validation_metrics=validation_metrics,
        overwrite=overwrite_artifacts,
        dataset_counts=dataset_counts,
        split_counts=actual_counts,
        artifact_directory=artifact_directory,
        verification_df=splits.test,
        expected_probabilities=test_probabilities,
    )

    # 보고서 저장
    save_stacking_test_report(
        test_df=splits.test,
        probabilities=test_probabilities,
        predictions=test_predictions,
        latencies_ms=test_latencies_ms,
        metrics=test_metrics,
        threshold=threshold,
        unavailable_models=test_unavailable,
        output_directory=report_directory,
        artifact_version=STACKING_ARTIFACT_VERSION,
    )

    print(f"[Stacking {STACKING_ARTIFACT_VERSION}] training completed")
    print(f"  train={len(splits.train)}")
    print(f"  validation={len(splits.validation)}")
    print(f"  test={len(splits.test)}")
    print(f"  holdout={len(holdout)}")
    print(f"  threshold={threshold:.6f}")
    print(f"  test_accuracy={test_metrics.accuracy:.4f}")
    print(f"  test_recall={test_metrics.recall:.4f}")
    print(f"  test_f2={test_metrics.f2:.4f}")
    print(f"  artifact={artifact_directory / 'model.joblib'}")

def main() -> None:
    """CLI 인자를 읽어 stacking 학습을 실행"""
    parser = argparse.ArgumentParser(
        description="Train the SafeFam stacking phishing classifier."
    )
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
    )
    parser.add_argument(
        "--max-false-positive-rate",
        type=float,
        default=MAX_NORMAL_FALSE_POSITIVE_RATE,
        help=(
            "정상 오탐 상한. 기본값은 채택 기준과 같으며, 완화하면 단독 운영 "
            "후보가 아닌 실험용 artifact가 만들어진다."
        ),
    )
    arguments = parser.parse_args()

    if arguments.max_false_positive_rate != MAX_NORMAL_FALSE_POSITIVE_RATE:
        print(
            "[Stacking] 경고: 정상 오탐 상한을 "
            f"{MAX_NORMAL_FALSE_POSITIVE_RATE}에서 "
            f"{arguments.max_false_positive_rate}로 완화했다. "
            "이 artifact는 단독 운영 후보가 아니다."
        )

    train_stacking(
        overwrite_artifacts=arguments.overwrite_artifacts,
        max_false_positive_rate=arguments.max_false_positive_rate,
    )

if __name__ == "__main__":
    main()
