"""Stacking 학습 스크립트의 임계값 선택 및 artifact 저장 테스트."""

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


def test_selects_threshold_using_validation_f2_and_recall() -> None:
    probabilities = np.asarray([0.1, 0.4, 0.8, 0.9])
    labels = pd.Series(["normal", "normal", "phishing", "phishing"])

    threshold, metrics = training.select_validation_threshold(
        probabilities,
        labels,
        target_recall=1.0,
    )

    assert 0.4 < threshold <= 0.8
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f2"] == pytest.approx(1.0)
    assert metrics["target_recall_met"] is True


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        (np.asarray([]), pd.Series(dtype=str), "must not be empty"),
        (
            np.asarray([0.1, np.nan]),
            pd.Series(["normal", "phishing"]),
            "finite",
        ),
        (
            np.asarray([0.1, 1.1]),
            pd.Series(["normal", "phishing"]),
            "between 0 and 1",
        ),
        (
            np.asarray([0.1]),
            pd.Series(["normal", "phishing"]),
            "same length",
        ),
    ],
)
def test_rejects_invalid_validation_inputs(
    probabilities: np.ndarray,
    labels: pd.Series,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        training.select_validation_threshold(probabilities, labels)


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
        )
