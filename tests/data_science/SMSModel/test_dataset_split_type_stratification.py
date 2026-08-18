"""세부 유형(type)이 split 배분 되는지 테스트"""
from __future__ import annotations

import pandas as pd

from data_science.SMSModel.dataset_splitting import (
    DatasetSplitConfig,
    split_grouped_dataset,
)

PHISHING_TYPE_GROUP_COUNTS = {
    "이벤트당첨사칭": 20,
    "금융기관사칭": 20,
    "지인가족사칭": 20,
}
NORMAL_TYPE_GROUP_COUNTS = {
    "일상대화": 30,
    "정상금융알림": 15,
}


def make_typed_dataset() -> pd.DataFrame:
    """유형이 여러 개인 그룹 단위 데이터셋 생성"""
    rows = []
    group_index = 0
    for label, type_counts in (
        ("phishing", PHISHING_TYPE_GROUP_COUNTS),
        ("normal", NORMAL_TYPE_GROUP_COUNTS),
    ):
        for type_name, group_count in type_counts.items():
            for _ in range(group_count):
                rows.append(
                    {
                        "text_fingerprint": f"fp-{group_index}",
                        "template_group_id": f"group-{group_index}",
                        "label": label,
                        "type": type_name,
                    }
                )
                group_index += 1
    return pd.DataFrame(rows)


def type_distribution_error(subset: pd.DataFrame, full: pd.DataFrame) -> float:
    """전체 대비 부분집합의 유형 분포 L1 거리(0~2). 작을수록 고르게 배분"""
    full_ratio = full["type"].value_counts(normalize=True)
    subset_ratio = subset["type"].value_counts(normalize=True)
    return sum(
        abs(full_ratio.get(name, 0.0) - subset_ratio.get(name, 0.0))
        for name in full_ratio.index
    )


def worst_type_error(splits, dataset: pd.DataFrame) -> float:
    """세 split 중 유형 분포가 가장 많이 어긋난 값을 반환"""
    return max(
        type_distribution_error(frame, dataset)
        for frame in (splits.train, splits.validation, splits.test)
    )


def test_type_stratification_reduces_distribution_error() -> None:
    """유형을 고려하면 유형 분포 편중이 기존 방식보다 나빠지지 않아야 함"""
    dataset = make_typed_dataset()

    baseline = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200, type_column=None),
    )
    stratified = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200),
    )

    assert worst_type_error(stratified, dataset) <= worst_type_error(baseline, dataset)


def test_type_weight_zero_matches_disabled_stratification() -> None:
    """type_weight=0은 유형을 보지 않는 것과 같은 결과를 내야 함"""
    dataset = make_typed_dataset()

    disabled = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200, type_column=None),
    )
    zero_weight = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200, type_weight=0.0),
    )

    assert set(disabled.test["text_fingerprint"]) == set(
        zero_weight.test["text_fingerprint"]
    )


def test_label_balance_is_preserved() -> None:
    """유형을 맞추느라 label 균형이 깨지지 않아야 함"""
    dataset = make_typed_dataset()

    splits = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200),
    )

    overall_phishing_ratio = (dataset["label"] == "phishing").mean()
    for frame in (splits.train, splits.validation, splits.test):
        ratio = (frame["label"] == "phishing").mean()
        assert abs(ratio - overall_phishing_ratio) < 0.15

def test_types_with_enough_groups_appear_in_every_split() -> None:
    """template group이 3개 이상인 유형은 세 split에 모두 나타나야 한다.

    group이 2개 이하인 유형은 group 무결성을 지키면서 세 split에 나눌 수 없으므로
    검증 대상에서 제외한다.
    """
    dataset = make_typed_dataset()

    splits = split_grouped_dataset(
        dataset,
        config=DatasetSplitConfig(candidate_count=200),
    )

    groups_per_type = dataset.groupby("type")["template_group_id"].nunique()
    splittable_types = set(groups_per_type[groups_per_type >= 3].index)

    for split_name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        missing = splittable_types - set(frame["type"])
        assert not missing, f"{split_name}에 없는 유형: {sorted(missing)}"
