"""Naive Bayes baseline 실행기의 orchestration 테스트."""

from types import SimpleNamespace

import pandas as pd

from data_science.SMSModel import run_naive_bayes_baseline as runner
from data_science.SMSModel.evaluation.evaluator import ModelEvaluationResult
from data_science.SMSModel.evaluation.latency import LatencyMetrics
from data_science.SMSModel.evaluation.metrics import ClassificationMetrics
from data_science.SMSModel.evaluation.threshold import ThresholdSelection


def make_result(model_name: str, threshold: float) -> ModelEvaluationResult:
    return ModelEvaluationResult(
        model_name=model_name,
        score_type="probability",
        selected_threshold=threshold,
        validation=ThresholdSelection(
            threshold=threshold,
            precision=1.0,
            recall=1.0,
            f2=1.0,
            false_negative_count=0,
            target_recall=0.96,
            target_recall_met=True,
        ),
        test_metrics=ClassificationMetrics(
            sample_count=2,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            f2=1.0,
            true_negative=1,
            false_positive=0,
            false_negative=0,
            true_positive=1,
        ),
        latency=LatencyMetrics(
            sample_count=1,
            warmup_count=0,
            average_ms=0.1,
            median_ms=0.1,
            p95_ms=0.1,
            minimum_ms=0.1,
            maximum_ms=0.1,
        ),
        metadata={"model_name": model_name},
    )


def test_runner_evaluates_both_variants_without_deploying_artifact(
    monkeypatch,
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "text": ["정상", "피싱 [URL]"],
            "text_norm": ["정상", "피싱 [URL]"],
            "has_url": [False, True],
            "label": ["normal", "phishing"],
        }
    )
    splits = SimpleNamespace(train=frame, validation=frame, test=frame)
    evaluated_models = []
    report_calls = []

    monkeypatch.setattr(runner, "load_data", lambda _path: (frame, frame.iloc[0:0]))
    monkeypatch.setattr(runner, "split_data", lambda _df: splits)

    def fake_evaluate(model, **kwargs):
        evaluated_models.append((model, kwargs))
        threshold = 0.35 if model.include_structural_features else 0.45
        return make_result(model.model_name, threshold)

    monkeypatch.setattr(runner, "train_and_evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "save_model_evaluation_reports",
        lambda results, **kwargs: report_calls.append((results, kwargs)),
    )
    monkeypatch.setattr(runner, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(runner, "VECTORIZER_PATH", tmp_path / "vectorizer.pkl")
    monkeypatch.setattr(runner, "NB_REPORT_DIRECTORY", tmp_path / "reports")

    results = runner.evaluate_naive_bayes_baselines()

    assert [model.include_structural_features for model, _ in evaluated_models] == [
        False,
        True,
    ]
    assert [result.model_name for result in results] == [
        "naive_bayes_text_only",
        "naive_bayes_structural",
    ]
    for _model, kwargs in evaluated_models:
        assert kwargs["train_df"] is frame
        assert kwargs["validation_df"] is frame
        assert kwargs["test_df"] is frame
    assert report_calls[0][0] == results
    assert report_calls[0][1]["output_directory"] == tmp_path / "reports"
