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

    # 임계값·경계 선정에 쓰는 split을 특정 출처 위주로 채우기 위한 설정.
    #
    # validation은 학습 pool에서 무작위로 뽑히는데, pool 정상의 60%가 증강
    # 데이터(synthetic_normal_v3, real_collected_v4)라 실제 수집 문자보다
    # 훨씬 쉬웠다. 그 결과 validation이 포화돼(정상 96.7~98.9%가 경계 아래)
    # 하이퍼파라미터 구성 간 차이가 표본 1건 수준으로 뭉개졌고, 실제로
    # validation에서 가장 좋아 보이던 설정이 판정셋에서는 더 나빴다(#102).
    #
    # 출처별 난이도(OOF 실측, 정상 평균 확률):
    #   synthetic_normal_v3 0.099 < real_collected_v4 0.073 < original 0.142
    #   판정셋(real_holdout) 0.179
    # original이 판정셋에 가장 가까워 선정용 split을 이쪽으로 몰아준다.
    selection_source_column: str | None = "source"
    selection_source: str | None = None
    selection_source_weight: float = 0.0

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
        if self.selection_source_weight < 0.0:
            raise ValueError(
                "selection_source_weight must not be negative"
            )