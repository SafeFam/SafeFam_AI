"""그룹 보존과 클래스 비율 최적화를 적용한 데이터 분할"""

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


def _label_distribution(
    df: pd.DataFrame,
    *,
    label_column: str,
    labels: list[str],
) -> np.ndarray:
    if df.empty:
        return np.zeros(len(labels), dtype=float)
    proportions = df[label_column].value_counts(normalize=True)
    return np.asarray(
        [proportions.get(label, 0.0) for label in labels],
        dtype=float,
    )


def _candidate_score(
    *,
    selected: pd.DataFrame,
    full_data: pd.DataFrame,
    target_size: float,
    label_column: str,
    labels: list[str],
) -> float:
    """목표 행 비율과 전체 클래스 비율에 가까울수록 낮은 점수 제공"""
    size_error = abs((len(selected) / len(full_data)) - target_size)
    class_error = np.abs(
        _label_distribution(
            full_data,
            label_column=label_column,
            labels=labels,
        )
        - _label_distribution(
            selected,
            label_column=label_column,
            labels=labels,
        )
    ).sum()
    return size_error + (class_error * 2.0)


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
    """여러 결정적 후보 중 크기와 클래스 비율이 가장 좋은 분할 선택"""
    labels = sorted(df[config.label_column].unique())
    required_labels = set(labels)
    n_groups = df[config.group_column].nunique()
    best: tuple[pd.DataFrame, pd.DataFrame] | None = None
    best_score = float("inf")

    for candidate_index in range(config.candidate_count):
        # GroupShuffleSplit의 test_size는 '행 수'가 아닌 '그룹 수' 비율로 작동
        # 그룹별 행 개수가 편중되어 있을 때 목표 행 비율을 충족할 수 있도록
        # 후보 탐색 시 test_size 그룹 비율에 약간의 변동을 부여
        if config.candidate_count > 1:
            scale = 0.8 + 0.4 * (candidate_index / (config.candidate_count - 1))
            candidate_test_size = selected_size * scale
            # 최소 1개 그룹, 최대 (n_groups - 1)개 그룹 범위 보장
            candidate_test_size = max(
                1 / n_groups,
                min((n_groups - 1) / n_groups, candidate_test_size),
            )
        else:
            candidate_test_size = selected_size

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=candidate_test_size,
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
