"""비교 실험용 피싱 분류 모델 artifact 저장 및 로딩 기능.

``joblib``은 pickle 기반이므로 역직렬화 전에 임의 코드가 실행될 수 있습니다.
따라서 이 모듈은 SafeFam이 직접 생성하고 관리하는 신뢰할 수 있는 로컬
artifact만 로드해야 합니다. SHA-256은 파일 손상과 잘못된 파일 조합을
검출하기 위한 checksum이며, 공격자가 만든 artifact를 인증하는 서명이 아닙니다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreType,
)

COMPARISON_ARTIFACT_SCHEMA_VERSION = 1

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

SHA256_HEX_LENGTH = 64

SUPPORTED_COMPARISON_MODELS = {
    "linear_svm_char_tfidf",
    "logistic_regression_morph_tfidf",
}


@dataclass(frozen=True)
class LoadedComparisonArtifact:
    """로드 및 검증이 완료된 비교 모델 artifact"""

    classifier: BasePhishingClassifier
    threshold: float
    metadata: dict[str, Any]


def _get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _collect_library_versions() -> dict[str, str]:
    """artifact 재현에 필요한 주요 실행환경 버전을 수집"""
    return {
        "python": platform.python_version(),
        "numpy": _get_package_version("numpy"),
        "pandas": _get_package_version("pandas"),
        "scikit_learn": _get_package_version("scikit-learn"),
        "joblib": _get_package_version("joblib"),
        "kiwipiepy": _get_package_version("kiwipiepy"),
    }


def _calculate_sha256(path: Path) -> str:
    """파일 전체를 읽어 SHA-256 checksum을 계산"""
    digest = hashlib.sha256()

    with path.open("rb") as artifact_file:
        for chunk in iter(
            lambda: artifact_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_dataset_fingerprint(
    dataset_fingerprint: str,
) -> None:
    """dataset fingerprint가 SHA-256 형식인지 확인"""
    if not isinstance(dataset_fingerprint, str):
        raise TypeError("dataset_fingerprint must be a string")

    if len(dataset_fingerprint) != SHA256_HEX_LENGTH:
        raise ValueError(
            "dataset_fingerprint must be a 64-character SHA-256 hex string"
        )

    try:
        int(dataset_fingerprint, 16)
    except ValueError as exc:
        raise ValueError(
            "dataset_fingerprint must contain only hexadecimal characters"
        ) from exc


def _validate_split_manifest_version(
    split_manifest_version: str,
) -> None:
    """split manifest 버전이 비어 있지 않은지 확인"""
    if not isinstance(split_manifest_version, str):
        raise TypeError("split_manifest_version must be a string")

    if not split_manifest_version.strip():
        raise ValueError("split_manifest_version must not be empty")


def _validate_sha256_hex(
    value: Any,
    *,
    field_name: str,
) -> None:
    """값이 64자리 SHA-256 hex 문자열인지 검증합니다."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError(
            f"{field_name} must be a 64-character SHA-256 hex string"
        )

    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must contain only hexadecimal characters"
        ) from exc


def _validate_supported_model_name(model_name: Any) -> str:
    """비교 artifact가 지원하는 모델 이름인지 검증합니다."""
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")

    if model_name not in SUPPORTED_COMPARISON_MODELS:
        raise ValueError(
            f"unsupported comparison model: {model_name}"
        )

    return model_name


def _validate_threshold(
    classifier: BasePhishingClassifier,
    threshold: float,
) -> float:
    """모델 score 유형에 맞는 threshold인지 검증"""
    numeric_threshold = float(threshold)

    if not np.isfinite(numeric_threshold):
        raise ValueError("threshold must be finite")

    if (
        classifier.score_type == ScoreType.PROBABILITY
        and not 0.0 <= numeric_threshold <= 1.0
    ):
        raise ValueError(
            "probability threshold must be between 0 and 1"
        )

    return numeric_threshold


def _get_estimator(classifier: BasePhishingClassifier) -> Any:
    """Adapter 내부의 scikit-learn estimator를 반환"""
    estimator = getattr(classifier, "model", None)

    if estimator is None:
        raise ValueError(
            "classifier does not contain a fitted model"
        )

    return estimator


def _get_vectorizer(classifier: BasePhishingClassifier) -> Any:
    """Adapter 내부의 fitted vectorizer를 반환"""
    vectorizer = getattr(classifier, "vectorizer", None)

    if vectorizer is None:
        raise ValueError(
            "classifier does not contain a vectorizer"
        )

    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "classifier vectorizer is not fitted"
        )

    return vectorizer


def _get_classes(
    classifier: BasePhishingClassifier,
) -> list[str]:
    """학습된 estimator의 클래스 목록을 반환"""
    estimator = _get_estimator(classifier)

    if not hasattr(estimator, "classes_"):
        raise ValueError(
            "classifier model does not contain fitted classes"
        )

    classes = [
        str(label)
        for label in estimator.classes_
    ]

    if set(classes) != {"normal", "phishing"}:
        raise ValueError(
            "classifier classes must contain normal and phishing"
        )

    if len(classes) != 2:
        raise ValueError(
            "classifier must be a binary classifier"
        )

    return classes


def _get_feature_count(
    classifier: BasePhishingClassifier,
) -> int:
    """vectorizer와 estimator의 feature 수가 일치하는지 검증"""
    estimator = _get_estimator(classifier)
    vectorizer = _get_vectorizer(classifier)

    if not hasattr(estimator, "n_features_in_"):
        raise ValueError(
            "classifier model does not expose n_features_in_"
        )

    vectorizer_feature_count = len(
        vectorizer.get_feature_names_out()
    )
    estimator_feature_count = int(
        estimator.n_features_in_
    )

    if vectorizer_feature_count != estimator_feature_count:
        raise ValueError(
            "model and vectorizer feature count mismatch: "
            f"model={estimator_feature_count}, "
            f"vectorizer={vectorizer_feature_count}"
        )

    return estimator_feature_count


def _validate_metadata_schema(
    metadata: dict[str, Any],
) -> None:
    """metadata.json의 필수 필드와 schema version을 검증"""
    if not isinstance(metadata, dict):
        raise TypeError(
            "artifact metadata must be a JSON object"
        )

    required_fields = {
        "schema_version",
        "artifact_id",
        "model_name",
        "score_type",
        "threshold",
        "classes",
        "feature_count",
        "dataset_fingerprint",
        "split_manifest_version",
        "created_at",
        "library_versions",
        "model_configuration",
        "model_sha256",
    }

    missing = required_fields - set(metadata)

    if missing:
        raise ValueError(
            f"artifact metadata is missing fields: {missing}"
        )

    if (
        metadata["schema_version"]
        != COMPARISON_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported comparison artifact schema version: "
            f"{metadata['schema_version']}"
        )

    artifact_id = metadata["artifact_id"]
    if not isinstance(artifact_id, str):
        raise TypeError("artifact_id must be a string")
    try:
        parsed_artifact_id = uuid.UUID(hex=artifact_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("artifact_id must be a UUID hex string") from exc
    if parsed_artifact_id.hex != artifact_id:
        raise ValueError("artifact_id must use canonical UUID hex format")

    _validate_supported_model_name(metadata["model_name"])

    if metadata["score_type"] not in {
        ScoreType.PROBABILITY.value,
        ScoreType.DECISION.value,
    }:
        raise ValueError("unsupported artifact score_type")

    threshold = metadata["threshold"]
    if isinstance(threshold, bool) or not isinstance(
        threshold,
        (int, float),
    ):
        raise TypeError("threshold must be a number")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")

    classes = metadata["classes"]
    if not isinstance(classes, list):
        raise TypeError("classes must be a list")
    if len(classes) != 2 or set(classes) != {"normal", "phishing"}:
        raise ValueError(
            "metadata classes must contain normal and phishing"
        )

    feature_count = metadata["feature_count"]
    if (
        isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count <= 0
    ):
        raise ValueError("feature_count must be a positive integer")

    _validate_dataset_fingerprint(metadata["dataset_fingerprint"])
    _validate_split_manifest_version(metadata["split_manifest_version"])
    _validate_sha256_hex(
        metadata["model_sha256"],
        field_name="model_sha256",
    )

    created_at = metadata["created_at"]
    if not isinstance(created_at, str):
        raise TypeError("created_at must be a string")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError("created_at must be a valid ISO datetime") from exc
    if (
        parsed_created_at.tzinfo is None
        or parsed_created_at.utcoffset() != timedelta(0)
    ):
        raise ValueError("created_at must include the UTC timezone")

    library_versions = metadata["library_versions"]
    if not isinstance(library_versions, dict):
        raise TypeError("library_versions must be a dictionary")
    required_libraries = {
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "joblib",
        "kiwipiepy",
    }
    if set(library_versions) != required_libraries:
        raise ValueError(
            "library_versions does not contain the required libraries"
        )
    if not all(
        isinstance(package_version, str)
        for package_version in library_versions.values()
    ):
        raise TypeError("library version values must be strings")

    if not isinstance(metadata["model_configuration"], dict):
        raise TypeError("model_configuration must be a dictionary")


def _validate_loaded_artifact(
    *,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> LoadedComparisonArtifact:
    """joblib payload와 metadata 조합이 서로 일치하는지 검증"""
    if not isinstance(payload, dict):
        raise TypeError(
            "artifact model payload must be a dictionary"
        )

    required_payload_fields = {
        "schema_version",
        "artifact_id",
        "model_name",
        "classifier",
        "threshold",
        "classes",
    }

    missing = required_payload_fields - set(payload)

    if missing:
        raise ValueError(
            f"artifact model payload is missing fields: {missing}"
        )

    if (
        payload["schema_version"]
        != COMPARISON_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported model payload schema version: "
            f"{payload['schema_version']}"
        )

    classifier = payload["classifier"]

    if not isinstance(
        classifier,
        BasePhishingClassifier,
    ):
        raise TypeError(
            "artifact classifier does not implement "
            "BasePhishingClassifier"
        )

    _validate_supported_model_name(classifier.model_name)

    if payload["artifact_id"] != metadata["artifact_id"]:
        raise ValueError(
            "model and metadata artifact IDs do not match"
        )

    if payload["model_name"] != metadata["model_name"]:
        raise ValueError(
            "model and metadata model names do not match"
        )

    if classifier.model_name != metadata["model_name"]:
        raise ValueError(
            "loaded classifier model name does not match metadata"
        )

    if classifier.score_type.value != metadata["score_type"]:
        raise ValueError(
            "loaded classifier score type does not match metadata"
        )

    payload_threshold = float(payload["threshold"])
    metadata_threshold = float(metadata["threshold"])

    if payload_threshold != metadata_threshold:
        raise ValueError(
            "model and metadata thresholds do not match"
        )

    validated_threshold = _validate_threshold(
        classifier,
        payload_threshold,
    )

    loaded_classes = _get_classes(classifier)

    if loaded_classes != list(payload["classes"]):
        raise ValueError(
            "loaded classifier classes do not match model payload"
        )

    if loaded_classes != list(metadata["classes"]):
        raise ValueError(
            "loaded classifier classes do not match metadata"
        )

    feature_count = _get_feature_count(classifier)

    if feature_count != int(metadata["feature_count"]):
        raise ValueError(
            "loaded classifier feature count does not match metadata"
        )

    if classifier.get_metadata() != metadata["model_configuration"]:
        raise ValueError(
            "loaded classifier configuration does not match metadata"
        )

    return LoadedComparisonArtifact(
        classifier=classifier,
        threshold=validated_threshold,
        metadata=metadata,
    )


def save_comparison_artifact(
    classifier: BasePhishingClassifier,
    *,
    threshold: float,
    output_directory: Path,
    dataset_fingerprint: str,
    split_manifest_version: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """학습된 LR 또는 Linear SVM을 비교 실험 artifact로 저장합니다."""
    if not isinstance(
        classifier,
        BasePhishingClassifier,
    ):
        raise TypeError(
            "classifier must implement BasePhishingClassifier"
        )

    _validate_supported_model_name(classifier.model_name)

    _validate_dataset_fingerprint(dataset_fingerprint)
    _validate_split_manifest_version(split_manifest_version)

    validated_threshold = _validate_threshold(
        classifier,
        threshold,
    )

    classes = _get_classes(classifier)
    feature_count = _get_feature_count(classifier)

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_directory / MODEL_FILENAME
    metadata_path = output_directory / METADATA_FILENAME

    if not overwrite and (
        model_path.exists() or metadata_path.exists()
    ):
        raise FileExistsError(
            "comparison artifact already exists; "
            "pass overwrite=True to replace it"
        )

    artifact_id = uuid.uuid4().hex

    payload = {
        "schema_version": COMPARISON_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "model_name": classifier.model_name,
        "classifier": classifier,
        "threshold": validated_threshold,
        "classes": classes,
    }

    temporary_model_path = (
        output_directory / f".model-{artifact_id}.tmp"
    )
    temporary_metadata_path = (
        output_directory / f".metadata-{artifact_id}.tmp"
    )

    try:
        joblib.dump(
            payload,
            temporary_model_path,
        )

        verification_payload = joblib.load(
            temporary_model_path
        )

        if (
            verification_payload.get("artifact_id")
            != artifact_id
        ):
            raise ValueError(
                "saved model artifact verification failed"
            )

        model_sha256 = _calculate_sha256(
            temporary_model_path
        )

        metadata: dict[str, Any] = {
            "schema_version": (
                COMPARISON_ARTIFACT_SCHEMA_VERSION
            ),
            "artifact_id": artifact_id,
            "model_name": classifier.model_name,
            "score_type": classifier.score_type.value,
            "threshold": validated_threshold,
            "classes": classes,
            "feature_count": feature_count,
            "dataset_fingerprint": dataset_fingerprint,
            "split_manifest_version": (
                split_manifest_version
            ),
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "library_versions": (
                _collect_library_versions()
            ),
            "model_configuration": (
                classifier.get_metadata()
            ),
            "model_sha256": model_sha256,
        }

        temporary_metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        os.replace(
            temporary_model_path,
            model_path,
        )
        os.replace(
            temporary_metadata_path,
            metadata_path,
        )

        return metadata

    finally:
        temporary_model_path.unlink(
            missing_ok=True
        )
        temporary_metadata_path.unlink(
            missing_ok=True
        )


def load_comparison_artifact(
    artifact_directory: Path,
) -> LoadedComparisonArtifact:
    """신뢰할 수 있는 로컬 비교 artifact를 로드하고 무결성을 검증합니다.

    외부에서 받은 경로나 사용자가 업로드한 joblib 파일에는 사용하면 안 됩니다.
    """
    artifact_directory = Path(artifact_directory)

    model_path = artifact_directory / MODEL_FILENAME
    metadata_path = artifact_directory / METADATA_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"comparison model artifact not found: {model_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"comparison metadata not found: {metadata_path}"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "comparison metadata is not valid JSON"
        ) from exc

    _validate_metadata_schema(metadata)

    actual_model_sha256 = _calculate_sha256(
        model_path
    )

    if actual_model_sha256 != metadata["model_sha256"]:
        raise ValueError(
            "comparison model checksum does not match metadata"
        )

    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(
            "comparison model artifact could not be loaded"
        ) from exc

    return _validate_loaded_artifact(
        payload=payload,
        metadata=metadata,
    )
