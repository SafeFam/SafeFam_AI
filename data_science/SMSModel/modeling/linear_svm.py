"""문자 n-gram TF-IDF 기반 Linear SVM 피싱 분류기"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)

# 문자 n-gram TF-IDF 설정
DEFAULT_NGRAM_RANGE = (3, 5)
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.95
DEFAULT_MAX_FEATURES = 30_000

# Linear SVM 설정
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_C = 1.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 10_000


class LinearSvmPhishingClassifier(BasePhishingClassifier):
    """문자 n-gram TF-IDF와 LinearSVC를 결합한 피싱 분류기 Adapter"""

    def __init__(self) -> None:
        
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=DEFAULT_NGRAM_RANGE,
            min_df=DEFAULT_MIN_DF,
            max_df=DEFAULT_MAX_DF,
            max_features=DEFAULT_MAX_FEATURES,
            lowercase=False,
            sublinear_tf=True,
        )

        self.model = LinearSVC(
            C=DEFAULT_C,
            class_weight=DEFAULT_CLASS_WEIGHT,
            random_state=DEFAULT_RANDOM_STATE,
            max_iter=DEFAULT_MAX_ITER,
            dual="auto",
        )

        self._is_fitted = False

    @property
    def model_name(self) -> str:
        """평가 보고서와 artifact에서 사용할 모델 이름."""
        return "linear_svm_char_tfidf"

    @property
    def score_type(self) -> ScoreType:
        """Linear SVM은 확률이 아닌 decision score를 반환"""
        return ScoreType.DECISION

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
            raise ValueError(f"missing Linear SVM columns: {missing}")

        if df.empty:
            raise ValueError("Linear SVM input DataFrame is empty")

        if df[list(required_columns)].isna().any().any():
            raise ValueError("Linear SVM input contains missing values")

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

    @staticmethod
    def _prepare_normalized_text(
        df: pd.DataFrame,
    ) -> pd.Series:
        """공통 전처리에서 생성된 text_norm을 모델 입력으로 반환"""
        return df["text_norm"].astype(str)

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> Self:
        """전달된 train split만 사용해 vectorizer와 SVM을 학습"""
        self._validate_dataframe(
            train_df,
            require_label=True,
        )

        normalized_text = self._prepare_normalized_text(train_df)

        features = self.vectorizer.fit_transform(normalized_text)
        labels = train_df["label"].astype(str)

        self.model.fit(features, labels)

        classes = list(self.model.classes_)

        if set(classes) != {"normal", "phishing"}:
            raise RuntimeError(
                "trained Linear SVM must contain normal and phishing classes"
            )

        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """학습되지 않은 vectorizer 또는 SVM의 사용을 차단"""
        if not self._is_fitted:
            raise RuntimeError("Linear SVM model is not fitted")

        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Linear SVM model classes are unavailable")

    def _orient_phishing_scores(
        self,
        raw_scores: np.ndarray,
    ) -> np.ndarray:
        """결정 점수가 클수록 phishing이 되도록 score 방향을 보정"""
        self._require_fitted()

        score_array = np.asarray(raw_scores, dtype=float)

        if score_array.ndim != 1:
            raise ValueError(
                "Linear SVM decision scores must be one-dimensional"
            )

        if not np.isfinite(score_array).all():
            raise ValueError(
                "Linear SVM decision scores must contain only finite values"
            )

        classes = list(self.model.classes_)

        if len(classes) != 2:
            raise RuntimeError(
                "Linear SVM must be a binary classifier"
            )

        if classes[1] == "phishing":
            return score_array

        if classes[0] == "phishing":
            return -score_array

        raise RuntimeError(
            "Linear SVM classes do not contain phishing"
        )

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> ScoreOutput:
        """각 메시지의 phishing 방향 decision score를 반환"""
        self._require_fitted()

        self._validate_dataframe(
            df,
            require_label=False,
        )

        normalized_text = self._prepare_normalized_text(df)

        # 추론에서는 fit_transform이 아니라 transform만 사용해야 합니다.
        features = self.vectorizer.transform(normalized_text)

        raw_scores = self.model.decision_function(features)

        phishing_scores = self._orient_phishing_scores(
            np.asarray(raw_scores, dtype=float)
        )

        return ScoreOutput(
            values=phishing_scores,
            score_type=self.score_type,
        )

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
            "vectorizer": {
                "type": "TfidfVectorizer",
                "analyzer": "char_wb",
                "ngram_range": list(DEFAULT_NGRAM_RANGE),
                "min_df": DEFAULT_MIN_DF,
                "max_df": DEFAULT_MAX_DF,
                "max_features": DEFAULT_MAX_FEATURES,
                "lowercase": False,
                "sublinear_tf": True,
            },
            "classifier": {
                "type": "LinearSVC",
                "C": DEFAULT_C,
                "class_weight": DEFAULT_CLASS_WEIGHT,
                "random_state": DEFAULT_RANDOM_STATE,
                "max_iter": DEFAULT_MAX_ITER,
                "dual": "auto",
            },
        }