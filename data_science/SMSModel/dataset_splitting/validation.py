"""데이터 split의 누수 및 무결성 검증."""

from __future__ import annotations

import pandas as pd

from .config import DatasetSplitConfig
from .splitter import DatasetSplits


def _named_parts(splits: DatasetSplits):
    return (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    )


def _validate_pairwise_disjoint(
    splits: DatasetSplits,
    *,
    column: str,
    description: str,
) -> None:
    parts = _named_parts(splits)
    for left_index, (left_name, left) in enumerate(parts):
        for right_name, right in parts[left_index + 1 :]:
            overlap = set(left[column].astype(str)) & set(right[column].astype(str))
            if overlap:
                raise ValueError(
                    f"{description} leakage between {left_name} and "
                    f"{right_name}: {sorted(overlap)[:10]}"
                )


def validate_dataset_splits(
    source: pd.DataFrame,
    splits: DatasetSplits,
    *,
    config: DatasetSplitConfig | None = None,
) -> None:
    """그룹·fingerprint 누수, 행 커버리지와 split 명칭을 모두 검증합니다."""
    config = config or DatasetSplitConfig()
    if source[config.fingerprint_column].duplicated().any():
        raise ValueError("source contains duplicate text fingerprints")

    for expected_name, part in _named_parts(splits):
        if part.empty:
            raise ValueError(f"{expected_name} split is empty")
        if "split" not in part.columns:
            raise ValueError(f"{expected_name} split column is missing")
        if set(part["split"].astype(str)) != {expected_name}:
            raise ValueError(f"invalid split name in {expected_name} data")
        if set(part[config.label_column].unique()) != set(
            source[config.label_column].unique()
        ):
            raise ValueError(f"{expected_name} does not contain every label")

    _validate_pairwise_disjoint(
        splits,
        column=config.group_column,
        description="template group",
    )
    _validate_pairwise_disjoint(
        splits,
        column=config.fingerprint_column,
        description="fingerprint",
    )

    combined = pd.concat(
        [splits.train, splits.validation, splits.test],
        ignore_index=True,
    )
    if len(combined) != len(source):
        raise ValueError(
            f"split row count mismatch: source={len(source)}, "
            f"split={len(combined)}"
        )
    if combined[config.fingerprint_column].duplicated().any():
        raise ValueError("a fingerprint appears more than once across splits")
    if set(source[config.fingerprint_column].astype(str)) != set(
        combined[config.fingerprint_column].astype(str)
    ):
        raise ValueError("split fingerprint coverage mismatch")
