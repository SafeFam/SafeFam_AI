"""Stacking 학습 스크립트의 artifact 저장 테스트"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel import run_stacking_training as training
from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)


def test_saves_and_reloads_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "stacking"
    model_path = artifact_directory / "model.joblib"
    metadata_path = artifact_directory / "metadata.json"

    monkeypatch.setattr(
        training,
        "STACKING_ARTIFACT_DIRECTORY",
        artifact_directory,
    )
    monkeypatch.setattr(training, "STACKING_MODEL_PATH", model_path)
    monkeypatch.setattr(training, "STACKING_METADATA_PATH", metadata_path)

    classifier = StackingPhishingClassifier(n_splits=2)
    monkeypatch.setattr(
        StackingPhishingClassifier,
        "get_metadata",
        lambda self: {
            "model_name": "stacking_phishing_classifier",
            "threshold": self.threshold,
        },
    )

    training.save_artifact(
        classifier,
        validation_metrics={
            "recall": 1.0,
            "f2": 1.0,
            "target_recall": 0.95,
            "target_recall_met": True,
        },
        overwrite=False,
        dataset_counts={
            "total_csv_rows": 3002,
            "training_pool_rows": 885,
            "holdout_rows": 210,
        },
        split_counts={"train": 623, "validation": 126, "test": 136},
    )

    payload = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert isinstance(payload["classifier"], StackingPhishingClassifier)
    assert metadata["model_sha256"] == training.calculate_sha256(model_path)
    assert not Path(metadata["dataset_path"]).is_absolute()
    serialized_metadata = json.dumps(metadata, ensure_ascii=False)
    assert '"text"' not in serialized_metadata


def test_does_not_overwrite_existing_artifact_without_permission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "stacking"
    artifact_directory.mkdir()
    model_path = artifact_directory / "model.joblib"
    model_path.touch()

    monkeypatch.setattr(
        training,
        "STACKING_ARTIFACT_DIRECTORY",
        artifact_directory,
    )
    monkeypatch.setattr(training, "STACKING_MODEL_PATH", model_path)

    classifier = StackingPhishingClassifier(n_splits=2)

    with pytest.raises(FileExistsError, match="--overwrite-artifacts"):
        training.save_artifact(
            classifier,
            validation_metrics={},
            overwrite=False,
            dataset_counts={
                "total_csv_rows": 3002,
                "training_pool_rows": 885,
                "holdout_rows": 210,
            },
            split_counts={"train": 623, "validation": 126, "test": 136},
        )


def test_metadata_records_measured_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """metadata가 상수 대신 실행 중 관측한 행 수를 기록하는지 검증"""
    artifact_directory = tmp_path / "stacking"
    model_path = artifact_directory / "model.joblib"
    metadata_path = artifact_directory / "metadata.json"

    monkeypatch.setattr(
        training,
        "STACKING_ARTIFACT_DIRECTORY",
        artifact_directory,
    )
    monkeypatch.setattr(training, "STACKING_MODEL_PATH", model_path)
    monkeypatch.setattr(training, "STACKING_METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        StackingPhishingClassifier,
        "get_metadata",
        lambda self: {"model_name": "stacking_phishing_classifier"},
    )

    measured_dataset_counts = {
        "total_csv_rows": 3100,
        "training_pool_rows": 900,
        "holdout_rows": 215,
    }
    measured_split_counts = {"train": 630, "validation": 130, "test": 140}

    training.save_artifact(
        StackingPhishingClassifier(n_splits=2),
        validation_metrics={},
        overwrite=False,
        dataset_counts=measured_dataset_counts,
        split_counts=measured_split_counts,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["artifact_version"] == training.STACKING_ARTIFACT_VERSION
    assert metadata["dataset"]["total_csv_rows"] == 3100
    assert metadata["dataset"]["training_pool_rows"] == 900
    assert metadata["dataset"]["holdout_rows"] == 215
    assert metadata["splits"] == measured_split_counts


def test_rejects_incomplete_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """관측 행 수가 누락되면 artifact를 저장하지 않는지 검증"""
    artifact_directory = tmp_path / "stacking"
    monkeypatch.setattr(
        training,
        "STACKING_ARTIFACT_DIRECTORY",
        artifact_directory,
    )
    monkeypatch.setattr(
        training, "STACKING_MODEL_PATH", artifact_directory / "model.joblib"
    )
    monkeypatch.setattr(
        training,
        "STACKING_METADATA_PATH",
        artifact_directory / "metadata.json",
    )

    with pytest.raises(ValueError, match="dataset_counts is missing keys"):
        training.save_artifact(
            StackingPhishingClassifier(n_splits=2),
            validation_metrics={},
            overwrite=False,
            dataset_counts={"total_csv_rows": 3002},
            split_counts={"train": 623, "validation": 126, "test": 136},
        )


def test_policy_run_uses_the_canonical_artifact_path() -> None:
    """상한을 지킨 실행만 정규 경로에 저장된다"""
    artifact, report = training.resolve_artifact_paths(
        training.MAX_NORMAL_FALSE_POSITIVE_RATE
    )

    assert artifact == training.STACKING_ARTIFACT_DIRECTORY
    assert report == training.STACKING_REPORT_DIRECTORY


def test_relaxed_run_is_kept_out_of_the_canonical_path() -> None:
    """완화한 실행이 정규 artifact를 덮어쓰면 안 된다

    분석 스크립트는 정규 경로를 기본으로 읽는다. 완화한 산출물이 거기에
    들어가면 단독 운영 후보가 아닌 artifact가 판정 대상이 된다.
    """
    relaxed = training.MAX_NORMAL_FALSE_POSITIVE_RATE + 0.04
    artifact, report = training.resolve_artifact_paths(relaxed)

    assert artifact != training.STACKING_ARTIFACT_DIRECTORY
    assert report != training.STACKING_REPORT_DIRECTORY
    assert artifact.name.endswith(training.EXPERIMENT_SUFFIX)
    assert report.name.endswith(training.EXPERIMENT_SUFFIX)
