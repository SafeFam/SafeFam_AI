"""그룹 보존과 클래스·유형 비율 최적화를 적용한 데이터 분할"""
from __future__ import annotations

from dataclasses import dataclass, replace

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


def _uncovered_value_weight(
    *,
    selected: pd.DataFrame,
    remaining: pd.DataFrame,
    full_data: pd.DataFrame,
    column: str,
    values: list[str],
) -> float:
    """양쪽 중 한 곳에라도 없는 값의 표본 비중 합을 반환한다(0~1).

    비율 오차만 보면 한 유형이 통째로 빠져도 다른 유형이 보정해 좋은 점수가
    나온다. 유형이 split에서 사라지는 것 자체에 비용을 매기되, 표본이 큰 유형이
    빠질수록 더 큰 비용이 되도록 비중으로 가중한다. 표본이 한두 건뿐이라 세
    split에 모두 넣을 수 없는 유형은 어떤 후보에서도 같은 값이라 순위에 영향을
    주지 않는다.
    """
    selected_values = set(selected[column].astype(str))
    remaining_values = set(remaining[column].astype(str))
    shares = full_data[column].astype(str).value_counts(normalize=True)

    return float(
        sum(
            shares.get(value, 0.0)
            for value in values
            if value not in selected_values or value not in remaining_values
        )
    )


def _candidate_score(
    *,
    selected: pd.DataFrame,
    remaining: pd.DataFrame,
    full_data: pd.DataFrame,
    target_size: float,
    label_column: str,
    labels: list[str],
    type_column: str | None,
    type_values: list[str],
    type_weight: float,
    selection_source_column: str | None = None,
    selection_source: str | None = None,
    selection_source_weight: float = 0.0,
) -> tuple[float, int, float, float]:
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
    uncovered_types = 0.0
    if type_column is not None and type_values and type_weight > 0.0:
        distribution_error += type_weight * _distribution_error(
            selected=selected,
            full_data=full_data,
            column=type_column,
            values=type_values,
        )
        uncovered_types = _uncovered_value_weight(
            selected=selected,
            remaining=remaining,
            full_data=full_data,
            column=type_column,
            values=type_values,
        )

    # 선정용 split은 지정 출처(실수집 데이터)로 채울수록 좋다. 유형 커버리지와
    # 크기를 먼저 맞춘 뒤, 같은 조건이면 지정 출처 비율이 높은 후보를 고른다.
    if (
        selection_source is not None
        and selection_source_column is not None
        and selection_source_column in selected.columns
        and selection_source_weight > 0.0
        and len(selected) > 0
    ):
        off_source_share = float(
            (
                selected[selection_source_column].astype(str)
                != selection_source
            ).mean()
        )
        distribution_error += selection_source_weight * off_source_share

    # 유형이 통째로 빠지는 것을 가장 먼저 막고, 그다음 크기와 분포를 맞춘다.
    # 크기는 1%p 단위 등급으로 비교해 미세한 차이로 후보가 뒤집히지 않게 한다.
    size_error_bucket = int(size_error / 0.01)
    return uncovered_types, size_error_bucket, distribution_error, size_error


def _contains_all_labels(
    df: pd.DataFrame,
    *,
    label_column: str,
    labels: set[str],
) -> bool:
    return set(df[label_column].unique()) == labels


def _stratified_group_candidates(
    df: pd.DataFrame,
    *,
    selected_size: float,
    config: DatasetSplitConfig,
    seed_offset: int,
    candidate_count: int,
) -> list[set[str]]:
    """유형별로 group을 비례 배분한 후보 집합들을 만든다.

    무작위 후보만으로는 표본이 적은 유형이 한쪽 split에서 통째로 빠지기 쉽다.
    유형마다 group을 섞어 비율만큼 떼어 두면 group이 둘 이상인 유형은 양쪽에
    반드시 남는다. 생성한 후보는 기존 후보와 같은 점수 기준으로 비교한다.
    """
    # type_weight가 0이면 유형을 고려하지 않는 설정이므로 후보도 만들지 않는다.
    if (
        config.type_column is None
        or config.type_weight == 0.0
        or config.type_column not in df.columns
    ):
        return []

    group_types = (
        df.groupby(config.group_column)[config.type_column].first().astype(str)
    )

    candidates: list[set[str]] = []
    for candidate_index in range(candidate_count):
        generator = np.random.default_rng(
            config.random_state + seed_offset + candidate_index
        )
        selected_groups: set[str] = set()

        for _, groups in group_types.groupby(group_types):
            group_ids = sorted(str(group_id) for group_id in groups.index)
            generator.shuffle(group_ids)

            take = int(round(len(group_ids) * selected_size))
            if len(group_ids) >= 2:
                # group이 둘 이상이면 양쪽에 최소 하나씩 남긴다.
                take = min(max(take, 1), len(group_ids) - 1)

            selected_groups.update(group_ids[:take])

        if selected_groups and len(selected_groups) < len(group_types):
            candidates.append(selected_groups)

    return candidates


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
    best_score = (float("inf"), int(1e9), float("inf"), float("inf"))

    feasible_group_counts = np.arange(1, n_groups)
    candidate_group_counts = np.resize(
        feasible_group_counts,
        max(config.candidate_count, len(feasible_group_counts)),
    )
    # 무작위 후보와 유형 비례 후보를 함께 평가한다.
    random_candidates: list[tuple[pd.DataFrame, pd.DataFrame]] = []
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
        random_candidates.append(
            (df.iloc[remaining_indices], df.iloc[selected_indices])
        )

    group_values = df[config.group_column].astype(str)
    stratified_candidates = [
        (
            df[~group_values.isin(selected_groups)],
            df[group_values.isin(selected_groups)],
        )
        for selected_groups in _stratified_group_candidates(
            df,
            selected_size=selected_size,
            config=config,
            seed_offset=random_state_offset,
            candidate_count=config.stratified_candidate_count,
        )
    ]

    for remaining, selected in random_candidates + stratified_candidates:

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
            remaining=remaining,
            full_data=df,
            target_size=selected_size,
            label_column=config.label_column,
            labels=labels,
            type_column=type_column,
            type_values=type_values,
            type_weight=config.type_weight,
            selection_source_column=config.selection_source_column,
            selection_source=config.selection_source,
            selection_source_weight=config.selection_source_weight,
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

    # test는 최종 평가용이라 전체 분포를 그대로 닮아야 한다 - 출처 선호를 걸지 않는다.
    train_validation, test = _select_best_group_split(
        df,
        selected_size=config.test_size,
        config=replace(
            config,
            selection_source=None,
            selection_source_weight=0.0,
        ),
        random_state_offset=0,
    )
    relative_validation_size = config.val_size / (config.train_size + config.val_size)
    # validation은 임계값·경계 선정에 쓰이므로 판정셋과 난이도가 비슷해야 한다.
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