"""형태소 TF-IDF 기반 Logistic Regression 피싱 분류기"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)
from data_science.SMSModel.tokenization.kiwi_tokenizer import kiwi_tokenize

# 모델 설정
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.95
DEFAULT_MAX_FEATURES = 20_000
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 1_000
DEFAULT_C = 1.0


class LogisticRegressionPhishingClassifier(BasePhishingClassifier):
    """Kiwi 형태소 TF-IDF와 Logistic Regression을 결합한 Adapter"""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=kiwi_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=DEFAULT_NGRAM_RANGE,
            min_df=DEFAULT_MIN_DF,
            max_df=DEFAULT_MAX_DF,
            max_features=DEFAULT_MAX_FEATURES,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            C=DEFAULT_C,
            class_weight=DEFAULT_CLASS_WEIGHT,
            random_state=DEFAULT_RANDOM_STATE,
            max_iter=DEFAULT_MAX_ITER,
            solver="liblinear",
        )
        self.classes_: list[str] | None = None
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        """평가 보고서와 artifact에서 사용할 모델 이름"""
        return "logistic_regression_morph_tfidf"

    @property
    def score_type(self) -> ScoreType:
        """Logistic Regression은 피싱 클래스 확률을 반환"""
        return ScoreType.PROBABILITY

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        *,
        require_label: bool,
    ) -> None:
        """학습 또는 추론에 필요한 입력 컬럼과 값을 검증"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")

        required_columns = {"text"}
        if require_label:
            required_columns.add("label")

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"missing Logistic Regression columns: {missing}")

        if df.empty:
            raise ValueError("Logistic Regression input DataFrame is empty")

        if df[list(required_columns)].isna().any().any():
            raise ValueError("Logistic Regression input contains missing values")

        if not df["text"].map(lambda value: isinstance(value, str)).all():
            raise TypeError("text values must be strings")

        if require_label:
            labels = set(df["label"].astype(str))
            supported_labels = {"normal", "phishing"}

            if not labels.issubset(supported_labels):
                raise ValueError(f"unsupported labels: {labels - supported_labels}")

            if labels != supported_labels:
                raise ValueError(
                    "training data must contain both normal and phishing labels"
                )

    @staticmethod
    def _prepare_normalized_text(df: pd.DataFrame) -> pd.Series:
        """API와 같은 공통 normalize_text 함수로 원문을 정규화"""
        return df["text"].map(normalize_text)

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> Self:
        """전달받은 train split만 사용해 vectorizer와 분류기 학습"""
        self._validate_dataframe(train_df, require_label=True)

        normalized_text = self._prepare_normalized_text(train_df)
        features = self.vectorizer.fit_transform(normalized_text)
        labels = train_df["label"].astype(str)

        self.model.fit(features, labels)
        self.classes_ = list(self.model.classes_)

        if "phishing" not in self.classes_:
            raise RuntimeError("trained model does not contain phishing class")

        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """학습되지 않았거나 불완전한 모델의 추론을 차단"""
        if not self._is_fitted or self.classes_ is None:
            raise RuntimeError("Logistic Regression model is not fitted")

    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        """각 메시지에 대한 phishing 클래스 확률을 반환"""
        self._require_fitted()
        self._validate_dataframe(df, require_label=False)

        normalized_text = self._prepare_normalized_text(df)
        features = self.vectorizer.transform(normalized_text)
        phishing_index = self.classes_.index("phishing")
        probabilities = self.model.predict_proba(features)[:, phishing_index]

        return ScoreOutput(
            values=np.asarray(probabilities, dtype=float),
            score_type=self.score_type,
        )

    def get_metadata(self) -> dict[str, Any]:
        """모델 재현과 artifact 추적에 필요한 전체 설정 반환"""
        return {
            **super().get_metadata(),
            "preprocessing": {
                "normalizer": "app.analysis.text.preprocessing.normalize_text",
                "tokenizer": (
                    "data_science.SMSModel.tokenization.kiwi_tokenizer.kiwi_tokenize"
                ),
            },
            "vectorizer": {
                "type": "TfidfVectorizer",
                "ngram_range": list(DEFAULT_NGRAM_RANGE),
                "min_df": DEFAULT_MIN_DF,
                "max_df": DEFAULT_MAX_DF,
                "max_features": DEFAULT_MAX_FEATURES,
                "lowercase": False,
                "sublinear_tf": True,
                "token_pattern": None,
            },
            "classifier": {
                "type": "LogisticRegression",
                "C": DEFAULT_C,
                "class_weight": DEFAULT_CLASS_WEIGHT,
                "random_state": DEFAULT_RANDOM_STATE,
                "max_iter": DEFAULT_MAX_ITER,
                "solver": "liblinear",
            },
        }
