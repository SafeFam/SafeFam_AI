"""SMS train/validation/test 분할 설정"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSplitConfig:
    """그룹 보존 분할을 동일하게 재현하기 위한 불변 설정"""

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    candidate_count: int = 500
    group_column: str = "template_group_id"
    label_column: str = "label"
    fingerprint_column: str = "text_fingerprint"

    type_column: str | None = "type"

    # 후보 점수에서 유형 분포 오차에 곱할 가중치
    type_weight: float = 1.0

    # 유형별로 group을 비례 배분해 만들 후보 수. 무작위 후보만으로는 표본이
    # 적은 유형이 한쪽 split에서 통째로 빠지기 쉬워 함께 평가한다.
    stratified_candidate_count: int = 200

    def __post_init__(self) -> None:
        total_size = self.train_size + self.val_size + self.test_size
        if abs(total_size - 1.0) > 1e-9:
            raise ValueError("train_size, val_size and test_size must sum to 1.0")
        for name, value in (
            ("train_size", self.train_size),
            ("val_size", self.val_size),
            ("test_size", self.test_size),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be greater than 0")
        if self.type_weight < 0.0:
            raise ValueError("type_weight must not be negative")
        if self.stratified_candidate_count < 0:
            raise ValueError(
                "stratified_candidate_count must not be negative"
            )