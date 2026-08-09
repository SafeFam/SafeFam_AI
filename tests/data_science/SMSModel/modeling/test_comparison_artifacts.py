"""비교 실험용 모델 artifact 저장 및 로딩 테스트"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.modeling.comparison_artifacts import (
    COMPARISON_ARTIFACT_SCHEMA_VERSION,
    METADATA_FILENAME,
    MODEL_FILENAME,
    load_comparison_artifact,
    save_comparison_artifact,
)
from data_science.SMSModel.modeling.linear_svm import (
    LinearSvmPhishingClassifier,
)
from data_science.SMSModel.modeling.logistic_regression import (
    LogisticRegressionPhishingClassifier,
)

TEST_DATASET_FINGERPRINT = "a" * 64
TEST_SPLIT_MANIFEST_VERSION = "sms_split_v1"


@pytest.fixture
def comparison_training_dataframe() -> pd.DataFrame:
    """두 비교 모델이 함께 사용할 균형 학습 데이터"""
    rows: list[dict[str, str]] = []

    for index in range(20):
        rows.append(
            {
                "text": (
                    f"오늘 가족 모임 시간을 안내합니다 {index}"
                ),
                "text_norm": (
                    f"오늘 가족 모임 시간을 안내합니다 {index}"
                ),
                "label": "normal",
            }
        )

        rows.append(
            {
                "text": (
                    "계좌가 정지되었습니다 즉시 인증하세요 "
                    f"https://bit.ly/fake{index}"
                ),
                "text_norm": (
                    "계좌가 정지되었습니다 즉시 인증하세요 [URL]"
                ),
                "label": "phishing",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def comparison_evaluation_dataframe() -> pd.DataFrame:
    """저장 전후 score를 비교할 평가 데이터"""
    return pd.DataFrame(
        {
            "text": [
                "오늘 저녁 가족 식사 안내",
                "계좌 정지 즉시 인증 https://danger.example",
                "[국외발신] 계좌 입금 요청",
            ],
            "text_norm": [
                "오늘 저녁 가족 식사 안내",
                "계좌 정지 즉시 인증 [URL]",
                "[국외발신] 계좌 입금 요청",
            ],
        }
    )


@pytest.fixture
def fitted_logistic_classifier(
    comparison_training_dataframe: pd.DataFrame,
) -> LogisticRegressionPhishingClassifier:
    """학습이 완료된 형태소 Logistic Regression"""
    return LogisticRegressionPhishingClassifier().fit(
        comparison_training_dataframe
    )


@pytest.fixture
def fitted_linear_svm_classifier(
    comparison_training_dataframe: pd.DataFrame,
) -> LinearSvmPhishingClassifier:
    """학습이 완료된 문자 n-gram Linear SVM"""
    return LinearSvmPhishingClassifier().fit(
        comparison_training_dataframe
    )


def _calculate_sha256(path: Path) -> str:
    """테스트에서 수정된 model.joblib의 checksum을 다시 계산"""
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        for chunk in iter(
            lambda: model_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


@pytest.mark.parametrize(
    ("fixture_name", "threshold"),
    [
        ("fitted_logistic_classifier", 0.40),
        ("fitted_linear_svm_classifier", -0.15),
    ],
)
def test_save_and_load_preserves_scores_and_predictions(
    request: pytest.FixtureRequest,
    fixture_name: str,
    threshold: float,
    comparison_evaluation_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """두 모델 모두 저장 전후 score와 prediction이 같아야 함"""
    classifier = request.getfixturevalue(fixture_name)

    artifact_directory = tmp_path / fixture_name

    expected_scores = classifier.predict_scores(
        comparison_evaluation_dataframe
    ).values

    expected_predictions = classifier.predict(
        comparison_evaluation_dataframe,
        threshold=threshold,
    )

    save_comparison_artifact(
        classifier,
        threshold=threshold,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    loaded = load_comparison_artifact(
        artifact_directory
    )

    actual_scores = loaded.classifier.predict_scores(
        comparison_evaluation_dataframe
    ).values

    actual_predictions = loaded.classifier.predict(
        comparison_evaluation_dataframe,
        threshold=loaded.threshold,
    )

    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_array_equal(
        actual_predictions,
        expected_predictions,
    )

    assert loaded.threshold == threshold
    assert (
        loaded.metadata["dataset_fingerprint"]
        == TEST_DATASET_FINGERPRINT
    )
    assert (
        loaded.metadata["split_manifest_version"]
        == TEST_SPLIT_MANIFEST_VERSION
    )


def test_saved_metadata_contains_reproducibility_information(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    tmp_path: Path,
) -> None:
    """metadata에 재현에 필요한 모든 정보가 포함"""
    artifact_directory = tmp_path / "logistic"

    metadata = save_comparison_artifact(
        fitted_logistic_classifier,
        threshold=0.50,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    assert (
        metadata["schema_version"]
        == COMPARISON_ARTIFACT_SCHEMA_VERSION
    )
    assert metadata["model_name"] == (
        "logistic_regression_morph_tfidf"
    )
    assert metadata["score_type"] == "probability"
    assert metadata["classes"] == [
        str(label)
        for label in fitted_logistic_classifier.model.classes_
    ]
    assert metadata["feature_count"] > 0
    assert metadata["created_at"]
    assert metadata["model_sha256"]

    assert set(
        metadata["library_versions"]
    ) == {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "joblib",
        "kiwipiepy",
    }

    assert (
        metadata["model_configuration"]
        == fitted_logistic_classifier.get_metadata()
    )


def test_load_rejects_unsupported_schema_version(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    tmp_path: Path,
) -> None:
    """지원하지 않는 metadata schema version은 거부"""
    artifact_directory = tmp_path / "invalid-schema"

    save_comparison_artifact(
        fitted_logistic_classifier,
        threshold=0.50,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    metadata_path = (
        artifact_directory / METADATA_FILENAME
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    metadata["schema_version"] = 999

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsupported comparison artifact schema",
    ):
        load_comparison_artifact(
            artifact_directory
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_message"),
    [
        ("dataset_fingerprint", "invalid", "64-character"),
        ("split_manifest_version", "", "must not be empty"),
        ("feature_count", 0, "positive integer"),
        ("classes", ["normal"], "normal and phishing"),
        ("created_at", "not-a-date", "valid ISO datetime"),
        ("model_sha256", "invalid", "64-character"),
    ],
)
def test_load_rejects_invalid_metadata_values(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    error_message: str,
) -> None:
    """필드는 존재하지만 값이 잘못된 metadata도 로드를 거부해야 함"""
    artifact_directory = tmp_path / field_name

    save_comparison_artifact(
        fitted_logistic_classifier,
        threshold=0.50,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    metadata_path = artifact_directory / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field_name] = invalid_value
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        (TypeError, ValueError),
        match=error_message,
    ):
        load_comparison_artifact(artifact_directory)


def test_load_rejects_model_vectorizer_feature_mismatch(
    fitted_linear_svm_classifier:
        LinearSvmPhishingClassifier,
    tmp_path: Path,
) -> None:
    """모델과 vectorizer의 feature 수가 다르면 로드를 거부."""
    artifact_directory = tmp_path / "feature-mismatch"

    save_comparison_artifact(
        fitted_linear_svm_classifier,
        threshold=0.0,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    model_path = artifact_directory / MODEL_FILENAME
    metadata_path = (
        artifact_directory / METADATA_FILENAME
    )

    payload = joblib.load(model_path)

    payload["classifier"].model.n_features_in_ += 1

    joblib.dump(
        payload,
        model_path,
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    metadata["model_sha256"] = _calculate_sha256(
        model_path
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="feature count mismatch",
    ):
        load_comparison_artifact(
            artifact_directory
        )


def test_load_rejects_mixed_model_and_metadata(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    fitted_linear_svm_classifier:
        LinearSvmPhishingClassifier,
    tmp_path: Path,
) -> None:
    """서로 다른 모델의 joblib과 metadata를 조합하면 거부 X"""
    logistic_directory = tmp_path / "logistic"
    svm_directory = tmp_path / "svm"

    save_comparison_artifact(
        fitted_logistic_classifier,
        threshold=0.50,
        output_directory=logistic_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    save_comparison_artifact(
        fitted_linear_svm_classifier,
        threshold=0.0,
        output_directory=svm_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    shutil.copyfile(
        svm_directory / MODEL_FILENAME,
        logistic_directory / MODEL_FILENAME,
    )

    with pytest.raises(
        ValueError,
        match="checksum does not match",
    ):
        load_comparison_artifact(
            logistic_directory
        )


def test_load_rejects_artifact_id_mismatch(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    fitted_linear_svm_classifier:
        LinearSvmPhishingClassifier,
    tmp_path: Path,
) -> None:
    """checksum이 갱신되어도 artifact ID가 다르면 조합을 거부 X"""
    logistic_directory = tmp_path / "logistic-id"
    svm_directory = tmp_path / "svm-id"

    save_comparison_artifact(
        fitted_logistic_classifier,
        threshold=0.50,
        output_directory=logistic_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    save_comparison_artifact(
        fitted_linear_svm_classifier,
        threshold=0.0,
        output_directory=svm_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    logistic_model_path = (
        logistic_directory / MODEL_FILENAME
    )
    logistic_metadata_path = (
        logistic_directory / METADATA_FILENAME
    )

    shutil.copyfile(
        svm_directory / MODEL_FILENAME,
        logistic_model_path,
    )

    metadata = json.loads(
        logistic_metadata_path.read_text(
            encoding="utf-8"
        )
    )
    metadata["model_sha256"] = _calculate_sha256(
        logistic_model_path
    )

    logistic_metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="artifact IDs do not match",
    ):
        load_comparison_artifact(
            logistic_directory
        )


def test_save_rejects_unfitted_classifier(
    tmp_path: Path,
) -> None:
    """학습하지 않은 모델은 artifact로 저장할 수 X"""
    classifier = LinearSvmPhishingClassifier()

    with pytest.raises(
        ValueError,
        match="not fitted|does not contain fitted classes",
    ):
        save_comparison_artifact(
            classifier,
            threshold=0.0,
            output_directory=tmp_path / "unfitted",
            dataset_fingerprint=TEST_DATASET_FINGERPRINT,
            split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
        )


def test_save_rejects_invalid_probability_threshold(
    fitted_logistic_classifier:
        LogisticRegressionPhishingClassifier,
    tmp_path: Path,
) -> None:
    """확률 모델에는 0~1 범위를 벗어난 threshold를 저장할 수 X"""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        save_comparison_artifact(
            fitted_logistic_classifier,
            threshold=1.50,
            output_directory=tmp_path / "invalid-threshold",
            dataset_fingerprint=TEST_DATASET_FINGERPRINT,
            split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
        )


def test_save_does_not_overwrite_existing_artifact_by_default(
    fitted_linear_svm_classifier:
        LinearSvmPhishingClassifier,
    tmp_path: Path,
) -> None:
    """명시하지 않으면 기존 비교 artifact를 덮어쓰지 X"""
    artifact_directory = tmp_path / "existing"

    save_comparison_artifact(
        fitted_linear_svm_classifier,
        threshold=0.0,
        output_directory=artifact_directory,
        dataset_fingerprint=TEST_DATASET_FINGERPRINT,
        split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        save_comparison_artifact(
            fitted_linear_svm_classifier,
            threshold=0.0,
            output_directory=artifact_directory,
            dataset_fingerprint=TEST_DATASET_FINGERPRINT,
            split_manifest_version=TEST_SPLIT_MANIFEST_VERSION,
        )
