"""SMS 피싱 분류 모델의 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ScoreType(str, Enum):
    """모델이 반환하는 피싱 점수의 의미."""

    PROBABILITY = "probability"
    DECISION = "decision"


@dataclass(frozen=True)
class ScoreOutput:
    """값이 클수록 피싱 가능성이 높은 1차원 점수 배열."""

    values: np.ndarray
    score_type: ScoreType

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 1:
            raise ValueError("score values must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("score values must contain only finite numbers")
        if self.score_type == ScoreType.PROBABILITY and (
            (values < 0.0) | (values > 1.0)
        ).any():
            raise ValueError("probability scores must be between 0 and 1")


class BasePhishingClassifier(ABC):
    """모든 SMS 피싱 모델 Adapter가 구현해야 하는 공통 계약."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """보고서와 artifact에서 사용할 모델 식별자."""

    @property
    @abstractmethod
    def score_type(self) -> ScoreType:
        """모델이 probability 또는 decision 중 무엇을 반환하는지 표시."""

    @property
    def default_threshold(self) -> float:
        """점수 유형에 맞는 기본 threshold."""
        return 0.5 if self.score_type == ScoreType.PROBABILITY else 0.0

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> BasePhishingClassifier:
        """train 데이터만 사용해 모델을 학습합니다."""

    @abstractmethod
    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        """각 행의 피싱 점수를 반환합니다."""

    @staticmethod
    def labels_from_scores(
        scores: np.ndarray,
        *,
        threshold: float,
    ) -> np.ndarray:
        """이미 계산한 score를 label로 변환해 test 중복 추론을 피합니다."""
        score_array = np.asarray(scores, dtype=float)
        if score_array.ndim != 1:
            raise ValueError("scores must be one-dimensional")
        if not np.isfinite(score_array).all():
            raise ValueError("scores must contain only finite numbers")
        return np.where(
            score_array >= threshold,
            "phishing",
            "normal",
        )

    def predict(
        self,
        df: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """기본 또는 지정 threshold를 적용해 label을 반환합니다."""
        selected_threshold = (
            self.default_threshold if threshold is None else threshold
        )
        scores = self.predict_scores(df)
        return self.labels_from_scores(
            scores.values,
            threshold=selected_threshold,
        )

    def get_metadata(self) -> dict[str, Any]:
        """artifact와 보고서에 기록할 공통 metadata."""
        return {
            "model_name": self.model_name,
            "score_type": self.score_type.value,
            "default_threshold": self.default_threshold,
        }
