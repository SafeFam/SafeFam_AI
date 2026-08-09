"""세 모델 통합 학습·평가 실행기 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from data_science.SMSModel import run_model_comparison as comparison


def test_output_directories_are_rooted_at_sms_model_workspace() -> None:
    """실행기 위치 때문에 modeling/ 아래로 결과가 저장되면 안 됩니다."""
    expected_root = Path(comparison.__file__).resolve().parent

    assert comparison.SMS_MODEL_DIR == expected_root
    assert comparison.COMPARISON_REPORT_DIRECTORY == (
        expected_root
        / "reports"
        / "model_evaluation"
        / "model_comparison"
    )
    assert comparison.COMPARISON_ARTIFACT_DIRECTORY == (
        expected_root / "artifacts" / "comparison"
    )


def test_missing_manifest_stops_before_any_split_is_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """고정 manifest가 없으면 split_data 호출 전에 실행을 중단합니다."""
    missing_manifest = tmp_path / "sms_split_v1.csv"
    split_called = False

    def fake_split_data(*args: Any, **kwargs: Any) -> None:
        nonlocal split_called
        split_called = True

    monkeypatch.setattr(
        comparison,
        "SPLIT_MANIFEST_PATH",
        missing_manifest,
    )
    monkeypatch.setattr(
        comparison,
        "split_data",
        fake_split_data,
    )

    with pytest.raises(
        FileNotFoundError,
        match="committed split manifest is required",
    ):
        comparison.run_model_comparison()

    assert split_called is False


def test_dataset_fingerprint_requires_successful_split_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """누수 검증을 통과하지 않은 split summary는 사용하지 않습니다."""
    summary_path = tmp_path / "dataset_split_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "a" * 64,
                "validation": {"passed": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        comparison,
        "DATASET_SPLIT_JSON_REPORT_PATH",
        summary_path,
    )

    with pytest.raises(
        ValueError,
        match="validation did not pass",
    ):
        comparison._load_dataset_fingerprint()


def test_all_models_receive_the_same_split_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모델마다 재분할하지 않고 같은 train/validation/test를 공유합니다."""
    train_df = pd.DataFrame({"split": ["train"]})
    validation_df = pd.DataFrame({"split": ["validation"]})
    test_df = pd.DataFrame({"split": ["test"]})

    models = [
        SimpleNamespace(model_name="model-a"),
        SimpleNamespace(model_name="model-b"),
        SimpleNamespace(model_name="model-c"),
    ]
    received_split_ids: list[tuple[int, int, int]] = []

    def fake_train_and_evaluate_model(
        model: Any,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_recall: float,
        latency_sample_count: int,
    ) -> Any:
        received_split_ids.append(
            (id(train_df), id(validation_df), id(test_df))
        )
        return SimpleNamespace(
            model_name=model.model_name,
            selected_threshold=0.0,
            test_metrics=SimpleNamespace(
                precision=1.0,
                recall=1.0,
                f1=1.0,
                f2=1.0,
                true_negative=1,
                false_positive=0,
                false_negative=0,
                true_positive=1,
            ),
            latency=SimpleNamespace(
                average_ms=1.0,
                p95_ms=1.0,
            ),
        )

    monkeypatch.setattr(
        comparison,
        "train_and_evaluate_model",
        fake_train_and_evaluate_model,
    )

    results = comparison._evaluate_models(
        models=models,  # type: ignore[arg-type]
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    expected_ids = (
        id(train_df),
        id(validation_df),
        id(test_df),
    )
    assert len(results) == 3
    assert received_split_ids == [expected_ids] * 3


def test_only_new_models_are_saved_as_comparison_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """공식 NB baseline은 기존 운영 artifact를 교체하지 않습니다."""
    models = [
        SimpleNamespace(model_name="naive_bayes_structural"),
        SimpleNamespace(
            model_name="logistic_regression_morph_tfidf"
        ),
        SimpleNamespace(model_name="linear_svm_char_tfidf"),
    ]
    results = [
        SimpleNamespace(
            model_name=model.model_name,
            selected_threshold=0.0,
        )
        for model in models
    ]
    saved_model_names: list[str] = []

    def fake_save_comparison_artifact(
        model: Any,
        **kwargs: Any,
    ) -> None:
        saved_model_names.append(model.model_name)

    monkeypatch.setattr(
        comparison,
        "save_comparison_artifact",
        fake_save_comparison_artifact,
    )

    comparison._save_new_model_artifacts(
        models=models,  # type: ignore[arg-type]
        results=results,  # type: ignore[arg-type]
        dataset_fingerprint="a" * 64,
        split_manifest_version="sms_split_v1",
        overwrite_artifacts=False,
    )

    assert saved_model_names == [
        "logistic_regression_morph_tfidf",
        "linear_svm_char_tfidf",
    ]


def test_run_report_uses_portable_paths_and_records_environment() -> None:
    """버전 관리되는 보고서에는 로컬 절대 경로를 기록하지 않습니다."""
    report = comparison._build_run_report(
        results=[],
        dataset_fingerprint="a" * 64,
        manifest_sha256="b" * 64,
        split_manifest_version="sms_split_v1",
        train_count=10,
        validation_count=2,
        test_count=2,
    )

    assert report["dataset"]["source_path"] == (
        "Data/SMSData/phishing_total_dataset_2705.csv"
    )
    assert report["split_manifest"]["path"] == (
        "splits/sms_split_v1.csv"
    )
    assert not Path(report["dataset"]["source_path"]).is_absolute()
    assert not Path(report["split_manifest"]["path"]).is_absolute()

    environment = report["execution_environment"]
    assert environment["python"]
    assert environment["platform"]
    assert environment["machine"]
    assert set(environment["library_versions"]) == {
        "numpy",
        "pandas",
        "scikit_learn",
        "kiwipiepy",
    }
