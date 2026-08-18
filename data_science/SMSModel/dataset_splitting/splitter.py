"""그룹 보존과 클래스·유형 비율 최적화를 적용한 데이터 분할"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import DatasetSplitConfig


@dataclass(frozen=True)
class DatasetSplits:
    """분할된 train, validation, test DataFrame 묶음"""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _column_distribution(
    df: pd.DataFrame,
    *,
    column: str,
    values: list[str],
) -> np.ndarray:
    """지정한 컬럼의 값 비율을 values 순서대로 반환"""
    if df.empty:
        return np.zeros(len(values), dtype=float)
    proportions = df[column].astype(str).value_counts(normalize=True)
    return np.asarray(
        [proportions.get(value, 0.0) for value in values],
        dtype=float,
    )


def _distribution_error(
    *,
    selected: pd.DataFrame,
    full_data: pd.DataFrame,
    column: str,
    values: list[str],
) -> float:
    """전체 분포와 선택된 부분 분포의 L1 거리를 반환(0~2)"""
    return float(
        np.abs(
            _column_distribution(full_data, column=column, values=values)
            - _column_distribution(selected, column=column, values=values)
        ).sum()
    )


def _candidate_score(
    *,
    selected: pd.DataFrame,
    full_data: pd.DataFrame,
    target_size: float,
    label_column: str,
    labels: list[str],
    type_column: str | None,
    type_values: list[str],
    type_weight: float,
) -> tuple[int, float, float]:
    """목표 행 비율과 클래스·유형 분포에 가까울수록 낮은 점수를 반환"""
    size_error = abs((len(selected) / len(full_data)) - target_size)

    distribution_error = _distribution_error(
        selected=selected,
        full_data=full_data,
        column=label_column,
        values=labels,
    )

    # 세부 유형이 한쪽 split에만 몰리면 그 split으로 고른 threshold가 다른 split에 전이 X
    # label 분포와 함께 유형 분포 오차도 반영
    if type_column is not None and type_values:
        distribution_error += type_weight * _distribution_error(
            selected=selected,
            full_data=full_data,
            column=type_column,
            values=type_values,
        )

    size_error_bucket = int(size_error / 0.01)
    return size_error_bucket, distribution_error, size_error


def _contains_all_labels(
    df: pd.DataFrame,
    *,
    label_column: str,
    labels: set[str],
) -> bool:
    return set(df[label_column].unique()) == labels


def _select_best_group_split(
    df: pd.DataFrame,
    *,
    selected_size: float,
    config: DatasetSplitConfig,
    random_state_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """여러 결정적 후보 중 크기와 클래스·유형 비율이 가장 좋은 분할 선택"""
    labels = sorted(df[config.label_column].astype(str).unique())
    required_labels = set(labels)

    # 유형 컬럼이 설정돼 있고 실제로 존재할 때만 유형 분포를 점수에 반영
    type_column = config.type_column
    type_values: list[str] = []
    if type_column is not None and type_column in df.columns:
        type_values = sorted(df[type_column].astype(str).unique())
    else:
        type_column = None

    n_groups = df[config.group_column].nunique()
    best: tuple[pd.DataFrame, pd.DataFrame] | None = None
    best_score = (int(1e9), float("inf"), float("inf"))

    feasible_group_counts = np.arange(1, n_groups)
    candidate_group_counts = np.resize(
        feasible_group_counts,
        max(config.candidate_count, len(feasible_group_counts)),
    )
    for candidate_index, selected_group_count in enumerate(candidate_group_counts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=int(selected_group_count),
            random_state=(config.random_state + random_state_offset + candidate_index),
        )
        remaining_indices, selected_indices = next(
            splitter.split(
                df,
                y=df[config.label_column],
                groups=df[config.group_column],
            )
        )
        remaining = df.iloc[remaining_indices]
        selected = df.iloc[selected_indices]

        if not _contains_all_labels(
            remaining,
            label_column=config.label_column,
            labels=required_labels,
        ) or not _contains_all_labels(
            selected,
            label_column=config.label_column,
            labels=required_labels,
        ):
            continue

        score = _candidate_score(
            selected=selected,
            full_data=df,
            target_size=selected_size,
            label_column=config.label_column,
            labels=labels,
            type_column=type_column,
            type_values=type_values,
            type_weight=config.type_weight,
        )
        if score < best_score:
            best_score = score
            best = (remaining, selected)

    if best is None:
        raise RuntimeError(
            "Could not create a grouped split containing every label. "
            "Inspect template groups and class distribution."
        )

    return tuple(part.reset_index(drop=True) for part in best)


def split_grouped_dataset(
    df: pd.DataFrame,
    *,
    config: DatasetSplitConfig | None = None,
) -> DatasetSplits:
    """동일 template_group_id를 보존하며 70/15/15에 가깝게 분할"""
    config = config or DatasetSplitConfig()
    required = {
        config.group_column,
        config.label_column,
        config.fingerprint_column,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing split columns: {missing}")
    if df.empty:
        raise ValueError("cannot split an empty dataset")
    if df[list(required)].isna().any().any():
        raise ValueError("split key columns contain missing values")
    if df[config.fingerprint_column].duplicated().any():
        raise ValueError("text_fingerprint must be unique before splitting")
    if df[config.group_column].nunique() < 3:
        raise ValueError("at least three template groups are required")

    train_validation, test = _select_best_group_split(
        df,
        selected_size=config.test_size,
        config=config,
        random_state_offset=0,
    )
    relative_validation_size = config.val_size / (config.train_size + config.val_size)
    train, validation = _select_best_group_split(
        train_validation,
        selected_size=relative_validation_size,
        config=config,
        random_state_offset=100_000,
    )

    parts = {"train": train, "validation": validation, "test": test}
    for split_name, part in parts.items():
        part["split"] = split_name

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )