"""모델 평가 JSON, CSV, Markdown 출력 테스트."""

import csv
import json

import pytest

from data_science.SMSModel.evaluation.evaluator import (
    train_and_evaluate_model,
)
from data_science.SMSModel.evaluation.reporting import (
    save_model_evaluation_reports,
)


def test_saves_json_csv_and_markdown_reports(
    tmp_path,
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
        latency_sample_count=1,
    )

    save_model_evaluation_reports([result], output_directory=tmp_path)

    json_path = tmp_path / "model_evaluation.json"
    csv_path = tmp_path / "model_evaluation.csv"
    markdown_path = tmp_path / "model_evaluation.md"
    assert json_path.exists() and csv_path.exists() and markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["models"][0]["model_name"] == "fake_probability"

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["false_negative"] == "0"
    assert rows[0]["recall"] == "1.0"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Phishing Model Evaluation" in markdown
    assert "fake_probability" in markdown


def test_rejects_empty_report_result_list(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        save_model_evaluation_reports([], output_directory=tmp_path)
