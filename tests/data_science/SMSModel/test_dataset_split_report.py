"""데이터 분할 통계 보고서 테스트."""

import json

import pandas as pd
import pytest

from data_science.SMSModel.dataset_splitting import (
    DatasetSplitConfig,
    DatasetSplits,
    split_grouped_dataset,
)
from data_science.SMSModel.reporting import (
    build_dataset_split_summary,
    calculate_dataset_fingerprint,
    generate_dataset_split_reports,
)
from data_science.SMSModel.template_grouping import TemplateGroupingConfig


@pytest.fixture
def report_data():
    rows = []
    message_types = ("기관사칭", "택배사칭", "정상알림톡")
    for group_index in range(60):
        label = "phishing" if group_index % 2 == 0 else "normal"
        for member_index in range(2 if group_index % 6 == 0 else 1):
            rows.append(
                {
                    "text_fingerprint": f"fp-{group_index}-{member_index}",
                    "template_group_id": f"group-{group_index}",
                    "label": label,
                    "type": message_types[group_index % len(message_types)],
                    "source": "original" if group_index % 3 else "user_added",
                }
            )
    source = pd.DataFrame(rows)
    split_config = DatasetSplitConfig(candidate_count=100)
    grouping_config = TemplateGroupingConfig(
        similarity_threshold=0.88,
        ngram_range=(2, 5),
    )
    splits = split_grouped_dataset(source, config=split_config)
    return source, splits, split_config, grouping_config


def test_dataset_fingerprint_is_reproducible(report_data):
    source, _, _, _ = report_data
    assert calculate_dataset_fingerprint(source) == calculate_dataset_fingerprint(
        source.copy()
    )


def test_dataset_fingerprint_ignores_row_order(report_data):
    source, _, _, _ = report_data
    shuffled = source.sample(frac=1.0, random_state=99).reset_index(drop=True)
    assert calculate_dataset_fingerprint(source) == calculate_dataset_fingerprint(
        shuffled
    )


def test_dataset_fingerprint_changes_when_label_changes(report_data):
    source, _, _, _ = report_data
    changed = source.copy()
    changed.loc[0, "label"] = "normal"
    assert calculate_dataset_fingerprint(source) != calculate_dataset_fingerprint(
        changed
    )


def test_summary_contains_counts_distributions_and_configuration(report_data):
    source, splits, split_config, grouping_config = report_data
    summary = build_dataset_split_summary(
        source,
        splits,
        split_config=split_config,
        grouping_config=grouping_config,
    )

    assert summary["dataset"]["row_count"] == len(source)
    assert summary["splits"]["train"]["row_count"] == len(splits.train)
    assert set(summary["splits"]["test"]["labels"]) == {
        "normal",
        "phishing",
    }
    assert summary["splits"]["validation"]["types"]
    assert summary["dataset"]["sources"]["original"]["count"] > 0
    assert "sources" in summary["splits"]["train"]
    assert summary["dataset"]["unresolved_other_phishing_count"] == 0
    assert summary["configuration"]["template_grouping"]["similarity_threshold"] == 0.88
    assert summary["validation"]["group_overlap_count"] == 0
    assert summary["validation"]["fingerprint_overlap_count"] == 0


def test_report_generation_creates_valid_json_and_markdown(
    tmp_path,
    report_data,
):
    source, splits, split_config, grouping_config = report_data
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"

    generate_dataset_split_reports(
        source,
        splits,
        split_config=split_config,
        grouping_config=grouping_config,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert parsed["validation"]["passed"] is True
    assert "## Split Overview" in markdown
    assert "## Message Type Distribution" in markdown
    assert "## Source Distribution" in markdown


def test_report_generation_is_deterministic(tmp_path, report_data):
    source, splits, split_config, grouping_config = report_data
    first_json = tmp_path / "first.json"
    first_md = tmp_path / "first.md"
    second_json = tmp_path / "second.json"
    second_md = tmp_path / "second.md"

    for json_path, markdown_path in (
        (first_json, first_md),
        (second_json, second_md),
    ):
        generate_dataset_split_reports(
            source,
            splits,
            split_config=split_config,
            grouping_config=grouping_config,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()


def test_report_generation_stops_on_group_leakage(tmp_path, report_data):
    source, splits, split_config, grouping_config = report_data
    invalid_splits = DatasetSplits(
        train=splits.train.copy(),
        validation=splits.validation.copy(),
        test=splits.test.copy(),
    )
    invalid_splits.validation.loc[0, "template_group_id"] = invalid_splits.train.iloc[
        0
    ]["template_group_id"]
    json_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"

    with pytest.raises(ValueError, match="template group leakage"):
        generate_dataset_split_reports(
            source,
            invalid_splits,
            split_config=split_config,
            grouping_config=grouping_config,
            json_path=json_path,
            markdown_path=markdown_path,
        )

    assert not json_path.exists()
    assert not markdown_path.exists()
