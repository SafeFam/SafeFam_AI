"""재현 가능한 SMS split manifest 저장 및 적용."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetSplitConfig
from .splitter import DatasetSplits
from .validation import validate_dataset_splits


MANIFEST_COLUMNS = [
    "text_fingerprint",
    "template_group_id",
    "split",
    "label",
    "type",
]
ALLOWED_SPLITS = {"train", "validation", "test"}


def build_split_manifest(splits: DatasetSplits) -> pd.DataFrame:
    """원문 없이 재현에 필요한 배정 정보만 정렬해 반환합니다."""
    combined = pd.concat(
        [splits.train, splits.validation, splits.test],
        ignore_index=True,
    )
    missing = set(MANIFEST_COLUMNS) - set(combined.columns)
    if missing:
        raise ValueError(f"cannot build manifest; missing columns: {missing}")
    return (
        combined[MANIFEST_COLUMNS]
        .sort_values(["split", "template_group_id", "text_fingerprint"])
        .reset_index(drop=True)
    )


def save_split_manifest(
    splits: DatasetSplits,
    path: Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """기존 파일은 명시적 요청 없이는 덮어쓰지 않고 CSV를 저장합니다."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"split manifest already exists: {path}. "
            "Use a new version or pass overwrite=True intentionally."
        )
    manifest = build_split_manifest(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return manifest


def apply_split_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    config: DatasetSplitConfig | None = None,
) -> DatasetSplits:
    """fingerprint를 키로 저장된 배정을 현재 데이터에 안전하게 적용합니다."""
    config = config or DatasetSplitConfig()
    required = {
        config.fingerprint_column,
        config.group_column,
        config.label_column,
        "split",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest columns: {missing}")
    if manifest[config.fingerprint_column].duplicated().any():
        raise ValueError("manifest contains duplicate text fingerprints")
    if set(manifest["split"].astype(str)) != ALLOWED_SPLITS:
        raise ValueError("manifest must contain only train, validation and test")
    if manifest.groupby(config.group_column)["split"].nunique().max() > 1:
        raise ValueError("manifest assigns one template group to multiple splits")

    current_keys = set(df[config.fingerprint_column].astype(str))
    manifest_keys = set(manifest[config.fingerprint_column].astype(str))
    if current_keys != manifest_keys:
        raise ValueError("dataset does not match split manifest fingerprints")

    # 현재 그룹 및 label이 manifest 생성 당시와 같은지도 확인합니다.
    verification = df[
        [config.fingerprint_column, config.group_column, config.label_column]
    ].merge(
        manifest[
            [config.fingerprint_column, config.group_column, config.label_column]
        ],
        on=config.fingerprint_column,
        suffixes=("_current", "_manifest"),
        validate="one_to_one",
    )
    for column in (config.group_column, config.label_column):
        if not (
            verification[f"{column}_current"].astype(str)
            == verification[f"{column}_manifest"].astype(str)
        ).all():
            raise ValueError(f"dataset {column} does not match split manifest")

    merged = df.merge(
        manifest[[config.fingerprint_column, "split"]],
        on=config.fingerprint_column,
        how="left",
        validate="one_to_one",
    )
    splits = DatasetSplits(
        train=merged[merged["split"] == "train"].copy().reset_index(drop=True),
        validation=merged[merged["split"] == "validation"].copy().reset_index(
            drop=True
        ),
        test=merged[merged["split"] == "test"].copy().reset_index(drop=True),
    )
    validate_dataset_splits(df, splits, config=config)
    return splits


def load_split_manifest(
    df: pd.DataFrame,
    path: Path,
    *,
    config: DatasetSplitConfig | None = None,
) -> DatasetSplits:
    """CSV manifest를 읽어 현재 데이터에 적용합니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split manifest does not exist: {path}")
    manifest = pd.read_csv(path, dtype="string")
    return apply_split_manifest(df, manifest, config=config)
