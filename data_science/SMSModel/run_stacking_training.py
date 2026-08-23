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
    select_probability_threshold_with_fallback,
)
from data_science.SMSModel.evaluation.adoption import AdoptionCriteria
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

STACKING_ARTIFACT_VERSION = "v9"

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

# 배포 이미지가 실제로 로드하는 경로. requirements.txt에 torch/transformers가
# 없으므로(#101) 인코더가 포함된 artifact는 여기 두면 로드에 실패한다.
PRODUCTION_ARTIFACT_DIRECTORY = STACKING_ARTIFACT_DIRECTORY.parent
PRODUCTION_REPORT_DIRECTORY = (
    SMS_MODEL_DIRECTORY / "reports" / "stacking_production"
)


def build_deployable_base_model_factories():
    """배포 이미지에서 실행 가능한 base model만 반환한다.

    인코더는 torch/transformers를 요구하는데 배포 requirements에 없다.
    프로덕션 artifact는 이 세 모델로만 만든다.
    """
    from data_science.SMSModel.modeling.linear_svm import (
        LinearSvmPhishingClassifier,
    )
    from data_science.SMSModel.modeling.logistic_regression import (
        LogisticRegressionPhishingClassifier,
    )
    from data_science.SMSModel.modeling.stacking import (
        _build_text_only_naive_bayes,
    )

    return {
        "naive_bayes": _build_text_only_naive_bayes,
        "logistic_regression": LogisticRegressionPhishingClassifier,
        "linear_svm": LinearSvmPhishingClassifier,
    }

# 단일 임계값 정책. 채택 기준과 재는 대상이 다르다 (#100 §2.4)
TARGET_RECALL = AdoptionCriteria().min_coverage_recall

# validation 정상 82건이 분해할 수 있는 하한은 3/82 = 0.0366이다.
# 사용자 대면 오탐 보장은 판정셋에서 alert_false_positive_rate가 담당한다.
MAX_NORMAL_FALSE_POSITIVE_RATE = 0.10
EXPECTED_DATASET_FINGERPRINT = (
    "500884cc4bd2ed9a45ce8b7dfbd16a"
    "ff6c7672422bbe7959f5b190fcdc98b1b3"
)

EXPECTED_TOTAL_CSV_ROWS = 3562
EXPECTED_TRAINING_POOL_ROWS = 949
EXPECTED_HOLDOUT_ROWS = 510
EXPECTED_SPLIT_COUNTS = {
    "train": 678,
    "validation": 136,
    "test": 135,
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
    """정책 상한을 지켰는지에 따라 artifact와 리포트 경로 선택"""
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
    validation_reference_metrics: dict[str, float] | None = None,
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
        "validation_reference": validation_reference_metrics,
        "dataset": {
            "total_csv_rows": dataset_counts["total_csv_rows"],
            "training_pool_rows": dataset_counts["training_pool_rows"],
            "holdout_rows": dataset_counts["holdout_rows"],
            "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        },
        "splits": dict(split_counts),
        "training_policy": {
            "training_split": "train",
            "threshold_selection_split": "train_oof",
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
    deployable: bool = False,
) -> None:
    """Stacking artifact를 학습하고 고정된 test split을 한 번 평가"""

    # committed v5 manifest가 없으면 실행 중단
    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: {SPLIT_MANIFEST_PATH}"
        )

    if SPLIT_MANIFEST_PATH.name != "sms_split_v6.csv":
        raise ValueError(
            "Stacking must use sms_split_v6.csv"
        )

    # 원본에서 학습 pool과 holdout을 분리. 실제 행 수는 아래에서 대조한다.
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

    # train 692건만 사용해 모델 학습.
    #
    # n_splits를 5에서 10으로 올렸다(#102). base model이 파인튜닝 인코더로
    # 바뀌면서 OOF가 최종 모델의 공정한 대리치가 아니게 됐기 때문이다 -
    # 5-fold면 fold 모델은 554건으로 학습하는데 최종 모델은 692건을 쓴다.
    # 얼린 인코더 위의 선형 분류기는 이 차이에 둔감했지만 파인튜닝은
    # 학습량에 민감해, OOF가 최종 모델을 과소평가하고 임계값이 필요보다
    # 훨씬 낮게 잡혔다(선정 normal_max 0.0597 vs 판정셋 필요값 약 0.42,
    # uncertain_normal_share 0.65). 10-fold면 fold 학습량이 623건으로 늘어
    # 격차가 줄어든다. base model 학습 횟수가 6회에서 11회로 늘어난다.
    classifier = StackingPhishingClassifier(
        n_splits=10,
        random_state=42,
        base_model_factories=(
            build_deployable_base_model_factories() if deployable else None
        ),
    )
    classifier.fit(splits.train)

    # threshold는 internal validation(수십~백여 건) 대신 train pool 전체
    # 규모의 메타 레벨 OOF 확률(classifier.oof_probabilities_)로 선정한다.
    # validation만으로 뽑으면 "오탐 0건" 같은 극값 통계라 표본이 몇 건만
    # 바뀌어도 경계가 크게 흔들렸다(#102 §5, 부트스트랩으로 확인:
    # phishing_min 5~95% 구간이 0.39~0.72까지 벌어짐). OOF는 표본이
    # 정상 기준 약 5배 커서 이 분산이 크게 줄어든다.
    threshold_selection, ceiling_was_relaxed = (
        select_probability_threshold_with_fallback(
            classifier.oof_probabilities_,
            classifier.oof_labels_,
            target_recall=TARGET_RECALL,
            max_false_positive_rate=max_false_positive_rate,
        )
    )

    # validation은 더 이상 threshold 선정에 쓰지 않지만, 진짜 안 보인
    # 데이터에서 이 threshold가 어떻게 나오는지 참고용으로는 계속 기록한다.
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
    if ceiling_was_relaxed:
        # 정책 상한(기본 0.10)은 데이터 구성이 바뀔 때마다 흔들린다 - split이
        # 달라지면 목표 recall에 필요한 오탐률이 상한을 넘나들어 매번 사람이
        # --max-false-positive-rate 값을 추측해 재실행해야 했다(#102 §5).
        # resolve_artifact_paths가 상수와 다른 값을 감지해 항상 -experiment
        # 경로로 저장하므로 "정책 후보가 아님"은 그대로 드러난다.
        max_false_positive_rate = threshold_selection.max_false_positive_rate
        print(
            f"[Stacking] 경고: 정상 오탐 상한을 {max_false_positive_rate:.4f}로 "
            "자동 완화했다 (train pool OOF에서 달성 가능한 최소치). "
            "이 artifact는 단독 운영 후보가 아니다."
        )
    threshold = threshold_selection.threshold
    validation_metrics = threshold_selection.to_validation_metrics()

    # 참고용 - 선정에는 안 쓰지만, 진짜 held-out인 validation에서 이
    # threshold가 어떻게 나오는지 기록해 OOF 기반 선정이 과최적화되지
    # 않았는지 확인할 수 있게 남긴다.
    validation_predictions = (validation_probabilities >= threshold).astype(int)
    validation_binary_labels = (
        splits.validation["label"].astype(str) == "phishing"
    ).astype(int)
    _val_is_phishing = validation_binary_labels == 1
    _val_is_normal = validation_binary_labels == 0
    validation_reference_metrics = {
        "recall": float(
            (validation_predictions[_val_is_phishing] == 1).sum()
            / max(int(_val_is_phishing.sum()), 1)
        ),
        "false_positive_rate": float(
            (validation_predictions[_val_is_normal] == 1).sum()
            / max(int(_val_is_normal.sum()), 1)
        ),
        "sample_count": int(len(splits.validation)),
        "normal_count": int(_val_is_normal.sum()),
    }

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

    if deployable:
        # 배포 경로에 바로 저장한다. 앱이 metadata.json의 model_sha256으로
        # 무결성을 검증하므로 save_artifact가 만든 metadata를 그대로 쓴다.
        artifact_directory = PRODUCTION_ARTIFACT_DIRECTORY
        report_directory = PRODUCTION_REPORT_DIRECTORY
    else:
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
        validation_reference_metrics=validation_reference_metrics,
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
        "--deployable",
        action="store_true",
        help=(
            "배포 이미지에서 실행 가능한 구성(인코더 제외)으로 학습해 "
            "프로덕션 경로(artifacts/stacking/)에 저장한다. requirements.txt에 "
            "torch/transformers가 없어 인코더 포함 artifact는 로드에 실패한다."
        ),
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
        deployable=arguments.deployable,
    )

if __name__ == "__main__":
    main()
