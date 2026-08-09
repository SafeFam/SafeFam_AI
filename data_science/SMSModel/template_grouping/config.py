"""SMS 템플릿 그룹화 설정"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_NGRAM_RANGE = (2, 5)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_FEATURES = 50_000


@dataclass(frozen=True)
class TemplateGroupingConfig:
    """동일한 입력과 설정으로 그룹화를 재현하기 위한 불변 설정"""

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
    min_df: int = DEFAULT_MIN_DF
    max_features: int | None = DEFAULT_MAX_FEATURES

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be greater than 0 and at most 1"
            )

        min_n, max_n = self.ngram_range
        if min_n <= 0 or max_n < min_n:
            raise ValueError(
                "ngram_range must contain positive values in ascending order"
            )
        if self.min_df <= 0:
            raise ValueError("min_df must be greater than 0")
        if self.max_features is not None and self.max_features <= 0:
            raise ValueError("max_features must be greater than 0")
