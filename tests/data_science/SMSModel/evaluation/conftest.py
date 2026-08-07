"""공통 평가기 테스트에서 사용하는 가짜 모델과 데이터."""

import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.modeling import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)


class FakeProbabilityClassifier(BasePhishingClassifier):
    """DataFrame의 mock_score를 그대로 반환하는 테스트 전용 모델."""

    def __init__(self) -> None:
        self.fitted = False
        self.fit_input: pd.DataFrame | None = None
        self.predict_call_count = 0

    @property
    def model_name(self) -> str:
        return "fake_probability"

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.PROBABILITY

    def fit(self, train_df: pd.DataFrame) -> BasePhishingClassifier:
        self.fitted = True
        self.fit_input = train_df.copy()
        return self

    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        if not self.fitted:
            raise RuntimeError("model is not fitted")

        self.predict_call_count += 1
        return ScoreOutput(
            values=df["mock_score"].to_numpy(dtype=float),
            score_type=self.score_type,
        )

    def get_metadata(self) -> dict[str, object]:
        return {
            **super().get_metadata(),
            "test_adapter": True,
        }


@pytest.fixture
def fake_model() -> FakeProbabilityClassifier:
    return FakeProbabilityClassifier()


@pytest.fixture
def evaluation_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "label": ["normal", "phishing", "normal", "phishing"],
            "mock_score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    validation = pd.DataFrame(
        {
            "label": ["normal", "normal", "phishing", "phishing"],
            "mock_score": [0.1, 0.4, 0.6, 0.9],
        }
    )
    test = pd.DataFrame(
        {
            "label": ["normal", "normal", "phishing", "phishing"],
            "mock_score": [0.2, 0.5, 0.7, 0.95],
        }
    )
    return train, validation, test
