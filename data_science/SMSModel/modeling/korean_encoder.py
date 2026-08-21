"""사전학습 한국어 인코더 임베딩 기반 피싱 분류기"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)

DEFAULT_MODEL_ID = "beomi/KcELECTRA-small-v2022"

# 벤치마크에서 p95 23.85ms, 절단 2.0%로 확인한 값
DEFAULT_MAX_LENGTH = 256

DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_C = 1.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 1_000


class KoreanEncoderPhishingClassifier(BasePhishingClassifier):
    """사전학습 인코더로 문장을 임베딩하고 선형 분류기를 얹은 Adapter"""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length

        self.model = LogisticRegression(
            C=DEFAULT_C,
            class_weight=DEFAULT_CLASS_WEIGHT,
            random_state=DEFAULT_RANDOM_STATE,
            max_iter=DEFAULT_MAX_ITER,
        )

        self._encoder = None
        self._tokenizer = None
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        """평가 보고서와 artifact에서 사용할 모델 이름"""
        return "korean_encoder_kcelectra"

    @property
    def score_type(self) -> ScoreType:
        """선형 분류기가 확률을 반환한다"""
        return ScoreType.PROBABILITY

    def _load_encoder(self) -> tuple[Any, Any]:
        """인코더와 토크나이저를 지연 로드. 최초 1회만 내려받기"""
        if self._encoder is None or self._tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._encoder = AutoModel.from_pretrained(self.model_id)
            self._encoder.eval()

        return self._encoder, self._tokenizer

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        *,
        require_label: bool,
    ) -> None:
        """학습 또는 추론에 필요한 DataFrame 구조와 값을 검증"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")

        required_columns = {"text_norm"}

        if require_label:
            required_columns.add("label")

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"missing Korean encoder columns: {missing}")

        if df.empty:
            raise ValueError("Korean encoder input DataFrame is empty")

        if df[list(required_columns)].isna().any().any():
            raise ValueError("Korean encoder input contains missing values")

        if not df["text_norm"].map(
            lambda value: isinstance(value, str)
        ).all():
            raise TypeError("text_norm values must be strings")

        if require_label:
            labels = set(df["label"].astype(str))
            supported_labels = {"normal", "phishing"}

            if not labels.issubset(supported_labels):
                raise ValueError(
                    f"unsupported labels: {labels - supported_labels}"
                )

            if labels != supported_labels:
                raise ValueError(
                    "training data must contain both normal and phishing labels"
                )

    def embed(self, texts: pd.Series) -> np.ndarray:
        """mean pooling 임베딩. 한 건씩 처리해 배치 불변을 지킨다

        묶어서 패딩하면 같은 문자라도 배치 구성에 따라 float32 누적 순서가
        달라져 1e-7 수준으로 흔들린다. 평가는 묶어서 하고 운영은 단건으로 하는
        구조라 그 차이가 그대로 재현성 문제가 된다. 짧은 문자가 대부분이라
        패딩 낭비가 커서 묶어도 빨라지지 않는다.
        """
        import torch

        encoder, tokenizer = self._load_encoder()
        vectors: list[np.ndarray] = []

        for value in texts:
            encoded = tokenizer(
                str(value),
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                hidden = encoder(**encoded).last_hidden_state

            vectors.append(hidden.mean(dim=1).numpy())

        return np.vstack(vectors).astype(np.float64)

    def fit(self, train_df: pd.DataFrame) -> Self:
        """전달된 train split만 사용해 학습"""
        self._validate_dataframe(train_df, require_label=True)

        features = self.embed(train_df["text_norm"])
        labels = train_df["label"].astype(str)

        self.model.fit(features, labels)

        classes = list(self.model.classes_)

        if set(classes) != {"normal", "phishing"}:
            raise RuntimeError(
                "trained Korean encoder must contain normal and phishing classes"
            )

        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """학습되지 않은 분류기의 사용을 차단"""
        if not self._is_fitted:
            raise RuntimeError("Korean encoder model is not fitted")

        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Korean encoder model classes are unavailable")

    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        """각 메시지의 phishing 확률을 반환"""
        self._require_fitted()
        self._validate_dataframe(df, require_label=False)

        features = self.embed(df["text_norm"])
        probabilities = self.model.predict_proba(features)

        classes = list(self.model.classes_)
        phishing_index = classes.index("phishing")

        return ScoreOutput(
            values=np.asarray(probabilities[:, phishing_index], dtype=float),
            score_type=self.score_type,
        )

    def __getstate__(self) -> dict[str, Any]:
        """인코더 가중치는 artifact에 넣지 않기"""
        state = self.__dict__.copy()
        state["_encoder"] = None
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """역직렬화 후 인코더는 최초 사용 시 다시 로드"""
        self.__dict__.update(state)

    def get_metadata(self) -> dict[str, Any]:
        """모델 재현과 artifact 추적에 필요한 설정을 반환"""
        return {
            **super().get_metadata(),
            "preprocessing": {
                "input_column": "text_norm",
                "normalization": (
                    "app.analysis.text.preprocessing.normalize_text"
                ),
            },
            "encoder": {
                "model_id": self.model_id,
                "max_length": self.max_length,
                "pooling": "mean",
                "weights_frozen": True,
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": DEFAULT_C,
                "class_weight": DEFAULT_CLASS_WEIGHT,
                "random_state": DEFAULT_RANDOM_STATE,
                "max_iter": DEFAULT_MAX_ITER,
            },
        }
