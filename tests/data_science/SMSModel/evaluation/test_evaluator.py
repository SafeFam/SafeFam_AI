"""공통 학습·validation 선택·test 평가 흐름 테스트."""

import pytest

from data_science.SMSModel.evaluation.evaluator import (
    train_and_evaluate_model,
)


def test_trains_selects_validation_threshold_and_evaluates_test(
    fake_model,
    evaluation_frames,
):
    train, validation, test = evaluation_frames

    result = train_and_evaluate_model(
        fake_model,
        train_df=train,
        validation_df=validation,
        test_df=test,
        target_recall=1.0,
        latency_sample_count=2,
    )

    assert fake_model.fitted is True
    assert fake_model.fit_input.equals(train)
    assert result.model_name == "fake_probability"
    assert result.score_type == "probability"
    assert result.selected_threshold == pytest.approx(0.6)
    assert result.validation.recall == pytest.approx(1.0)
    assert result.test_metrics.true_negative == 2
    assert result.test_metrics.true_positive == 2
    assert result.test_metrics.false_negative == 0
    assert result.metadata["test_adapter"] is True
    assert result.latency.sample_count == 2
    assert result.to_dict()["test_metrics"]["recall"] == pytest.approx(1.0)


def test_test_labels_do_not_change_selected_validation_threshold(
    fake_model,
    evaluation_frames,
):
    train, validation, test = evaluation_frames
    first = train_and_evaluate_model(
        fake_model,
        train_df=train,
        validation_df=validation,
        test_df=test,
        target_recall=1.0,
        latency_sample_count=1,
    )

    changed_test = test.copy()
    changed_test["label"] = list(reversed(changed_test["label"].tolist()))
    second_model = type(fake_model)()
    second = train_and_evaluate_model(
        second_model,
        train_df=train,
        validation_df=validation,
        test_df=changed_test,
        target_recall=1.0,
        latency_sample_count=1,
    )

    assert first.selected_threshold == second.selected_threshold
