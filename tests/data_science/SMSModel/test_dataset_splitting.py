"""그룹 보존 split과 manifest의 단위 테스트."""

import pandas as pd
import pytest

from data_science.SMSModel.dataset_splitting import (
    DatasetSplitConfig,
    load_split_manifest,
    save_split_manifest,
    split_grouped_dataset,
    validate_dataset_splits,
)


def make_dataset(group_count: int = 60) -> pd.DataFrame:
    rows = []
    for group_index in range(group_count):
        label = "phishing" if group_index % 2 == 0 else "normal"
        # 일부 그룹은 두 행을 가져 그룹 단위 분할을 실제로 검증합니다.
        member_count = 2 if group_index % 5 == 0 else 1
        for member_index in range(member_count):
            rows.append(
                {
                    "text_fingerprint": f"fp-{group_index}-{member_index}",
                    "template_group_id": f"group-{group_index}",
                    "label": label,
                    "type": "test-type",
                }
            )
    return pd.DataFrame(rows)


def fingerprint_sets(splits):
    return {
        "train": set(splits.train["text_fingerprint"]),
        "validation": set(splits.validation["text_fingerprint"]),
        "test": set(splits.test["text_fingerprint"]),
    }


def test_same_template_group_stays_in_one_split():
    splits = split_grouped_dataset(make_dataset())
    combined = pd.concat([splits.train, splits.validation, splits.test])
    assert combined.groupby("template_group_id")["split"].nunique().max() == 1


def test_split_is_reproducible_with_same_seed():
    df = make_dataset()
    first = split_grouped_dataset(df, config=DatasetSplitConfig(random_state=42))
    second = split_grouped_dataset(df, config=DatasetSplitConfig(random_state=42))
    assert fingerprint_sets(first) == fingerprint_sets(second)


def test_split_ratios_are_close_to_targets():
    df = make_dataset()
    splits = split_grouped_dataset(df)
    assert abs(len(splits.train) / len(df) - 0.70) <= 0.05
    assert abs(len(splits.validation) / len(df) - 0.15) <= 0.05
    assert abs(len(splits.test) / len(df) - 0.15) <= 0.05


def test_each_split_contains_both_labels():
    splits = split_grouped_dataset(make_dataset())
    for part in (splits.train, splits.validation, splits.test):
        assert set(part["label"]) == {"normal", "phishing"}


def test_each_split_label_ratio_is_close_to_full_dataset():
    df = make_dataset()
    expected = df["label"].value_counts(normalize=True)
    splits = split_grouped_dataset(df)

    for part in (splits.train, splits.validation, splits.test):
        actual = part["label"].value_counts(normalize=True)
        for label in expected.index:
            assert abs(actual[label] - expected[label]) <= 0.10


def test_validation_rejects_group_leakage():
    df = make_dataset()
    splits = split_grouped_dataset(df)
    splits.validation.loc[0, "template_group_id"] = splits.train.iloc[0][
        "template_group_id"
    ]
    with pytest.raises(ValueError, match="template group leakage"):
        validate_dataset_splits(df, splits)


def test_manifest_round_trip(tmp_path):
    df = make_dataset()
    expected = split_grouped_dataset(df)
    path = tmp_path / "split.csv"
    save_split_manifest(expected, path)
    actual = load_split_manifest(df, path)
    assert fingerprint_sets(actual) == fingerprint_sets(expected)


def test_manifest_rejects_changed_dataset(tmp_path):
    df = make_dataset()
    path = tmp_path / "split.csv"
    save_split_manifest(split_grouped_dataset(df), path)
    changed = df.iloc[:-1].copy()
    with pytest.raises(ValueError, match="does not match"):
        load_split_manifest(changed, path)


def test_manifest_rejects_changed_group_assignment(tmp_path):
    df = make_dataset()
    path = tmp_path / "split.csv"
    save_split_manifest(split_grouped_dataset(df), path)
    changed = df.copy()
    changed.loc[0, "template_group_id"] = "changed-group"
    with pytest.raises(ValueError, match="template_group_id"):
        load_split_manifest(changed, path)


def test_manifest_does_not_overwrite_by_default(tmp_path):
    splits = split_grouped_dataset(make_dataset())
    path = tmp_path / "split.csv"
    save_split_manifest(splits, path)
    with pytest.raises(FileExistsError):
        save_split_manifest(splits, path)


def test_manifest_round_trip_with_custom_column_names(tmp_path):
    config = DatasetSplitConfig(
        group_column="group_key",
        label_column="target",
        fingerprint_column="fingerprint_key",
    )
    df = make_dataset().rename(
        columns={
            "template_group_id": config.group_column,
            "label": config.label_column,
            "text_fingerprint": config.fingerprint_column,
        }
    )
    expected = split_grouped_dataset(df, config=config)
    path = tmp_path / "custom-split.csv"

    save_split_manifest(expected, path, config=config)
    actual = load_split_manifest(df, path, config=config)

    for split_name in ("train", "validation", "test"):
        expected_keys = set(getattr(expected, split_name)[config.fingerprint_column])
        actual_keys = set(getattr(actual, split_name)[config.fingerprint_column])
        assert actual_keys == expected_keys


def test_unequal_group_sizes_are_optimized_for_row_ratios():
    rows = []
    group_sizes = [30, 20, 12, 10, 8, 6, 4, 4, 3, 3, 2, 2]
    for group_index, group_size in enumerate(group_sizes):
        label = "phishing" if group_index % 2 == 0 else "normal"
        for member_index in range(group_size):
            rows.append(
                {
                    "text_fingerprint": f"unequal-{group_index}-{member_index}",
                    "template_group_id": f"unequal-group-{group_index}",
                    "label": label,
                    "type": "test-type",
                }
            )
    df = pd.DataFrame(rows)

    splits = split_grouped_dataset(df)

    assert abs(len(splits.test) / len(df) - 0.15) <= 0.05
    assert abs(len(splits.validation) / len(df) - 0.15) <= 0.05
