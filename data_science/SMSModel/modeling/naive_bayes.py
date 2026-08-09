"""기존 SafeFam Naive Bayes 모델의 공통 평가기 Adapter"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, spmatrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB

from app.analysis.text.preprocessing import (
    extract_struct_feature_matrix,
)
from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)

# 기존 운영 NB 설정을 그대로 유지
DEFAULT_NB_ALPHA = 0.01
DEFAULT_CALIBRATION_METHOD = "isotonic"
DEFAULT_CALIBRATION_CV = 5

DEFAULT_NGRAM_RANGE = (2, 4)
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.95
DEFAULT_MAX_FEATURES = 8_000


class NaiveBayesPhishingClassifier(BasePhishingClassifier):
    """Calibrated ComplementNB 기반 피싱 분류기 Adapter"""

    def __init__(
        self,
        *,
        include_structural_features: bool,
        alpha: float = DEFAULT_NB_ALPHA,
        calibration_method: str = DEFAULT_CALIBRATION_METHOD,
        calibration_cv: int = DEFAULT_CALIBRATION_CV,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        if calibration_cv < 2:
            raise ValueError("calibration_cv must be at least 2")

        self.include_structural_features = include_structural_features
        self.alpha = alpha
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv

        self.vectorizer = self._build_vectorizer()
        self.model: CalibratedClassifierCV | None = None
        self.classes_: list[str] | None = None

    @property
    def model_name(self) -> str:
        """보고서에서 두 NB 실험을 구분할 이름을 반환"""
        if self.include_structural_features:
            return "naive_bayes_structural"

        return "naive_bayes_text_only"

    @property
    def score_type(self) -> ScoreType:
        """CalibratedClassifierCV를 사용하므로 0~1 확률 반환"""
        return ScoreType.PROBABILITY

    @staticmethod
    def _build_vectorizer() -> CountVectorizer:
        """기존 운영 모델과 동일한 문자 n-gram CountVectorizer 생성"""
        return CountVectorizer(
            analyzer="char_wb",
            ngram_range=DEFAULT_NGRAM_RANGE,
            min_df=DEFAULT_MIN_DF,
            max_df=DEFAULT_MAX_DF,
            max_features=DEFAULT_MAX_FEATURES,
        )

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        *,
        require_label: bool,
        calibration_cv: int | None = None,
    ) -> None:
        """학습 또는 추론에 필요한 DataFrame 컬럼 및 샘플 개수 검사"""
        required_columns = {
            "text",
            "text_norm",
            "has_url",
        }

        if require_label:
            required_columns.add("label")

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(f"missing Naive Bayes columns: {missing}")

        if df.empty:
            raise ValueError("Naive Bayes input DataFrame is empty")

        if df[list(required_columns)].isna().any().any():
            raise ValueError("Naive Bayes input contains missing values")

        if require_label:
            labels = set(df["label"].astype(str))

            if not labels.issubset({"normal", "phishing"}):
                raise ValueError(
                    f"unsupported labels: {labels - {'normal', 'phishing'}}"
                )

            if labels != {"normal", "phishing"}:
                raise ValueError(
                    "training data must contain both normal and phishing labels"
                )

            # CalibratedClassifierCV의 폴드(CV) 수 대비 클래스별 최소 샘플 수 검증
            if calibration_cv is not None:
                min_class_count = df["label"].value_counts().min()
                if min_class_count < calibration_cv:
                    raise ValueError(
                        f"each class must have at least {calibration_cv} samples for calibration CV, "
                        f"but smallest class has {min_class_count} samples"
                    )

    def _prepare_normalized_text(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """text_norm을 사용하되 공통 전처리와 일치하는지 보장"""
        return df["text_norm"].astype(str)

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
        *,
        fit_vectorizer: bool,
    ) -> spmatrix:
        """모델 모드에 맞는 sparse feature matrix 생성"""
        text_norm = self._prepare_normalized_text(df)

        if fit_vectorizer:
            text_features = self.vectorizer.fit_transform(text_norm)
        else:
            text_features = self.vectorizer.transform(text_norm)

        if not self.include_structural_features:
            return text_features.tocsr()

        structural_features = extract_struct_feature_matrix(
            df["text"],
            df["has_url"],
        )

        return hstack(
            [
                text_features,
                csr_matrix(structural_features),
            ],
            format="csr",
        )

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> BasePhishingClassifier:
        """train split만 사용해 ComplementNB와 확률 보정기 학습"""
        self._validate_dataframe(
            train_df,
            require_label=True,
            calibration_cv=self.calibration_cv,
        )

        feature_matrix = self._build_feature_matrix(
            train_df,
            fit_vectorizer=True,
        )

        base_model = ComplementNB(
            alpha=self.alpha,
        )

        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=self.calibration_method,
            cv=self.calibration_cv,
        )

        calibrated_model.fit(
            feature_matrix,
            train_df["label"].astype(str),
        )

        self.model = calibrated_model
        self.classes_ = list(calibrated_model.classes_)

        if "phishing" not in self.classes_:
            raise RuntimeError("trained model does not contain phishing class")

        return self

    def _require_fitted(self) -> None:
        """학습되지 않은 Adapter의 추론 차단"""
        if self.model is None or self.classes_ is None:
            raise RuntimeError("Naive Bayes model is not fitted")

    def predict_scores(
        self,
        df: pd.DataFrame,
    ) -> ScoreOutput:
        """각 메시지의 보정된 phishing 확률 반환"""
        self._require_fitted()

        self._validate_dataframe(
            df,
            require_label=False,
        )

        feature_matrix = self._build_feature_matrix(
            df,
            fit_vectorizer=False,
        )

        phishing_index = self.classes_.index("phishing")

        probabilities = self.model.predict_proba(feature_matrix)[:, phishing_index]

        return ScoreOutput(
            values=np.asarray(
                probabilities,
                dtype=float,
            ),
            score_type=self.score_type,
        )

    def get_metadata(self) -> dict[str, Any]:
        """평가 보고서와 artifact 추적에 필요한 NB 설정을 반환"""
        return {
            **super().get_metadata(),
            "classifier": "ComplementNB",
            "alpha": self.alpha,
            "calibration_method": (self.calibration_method),
            "calibration_cv": self.calibration_cv,
            "include_structural_features": (self.include_structural_features),
            "vectorizer": {
                "type": "CountVectorizer",
                "analyzer": "char_wb",
                "ngram_range": list(DEFAULT_NGRAM_RANGE),
                "min_df": DEFAULT_MIN_DF,
                "max_df": DEFAULT_MAX_DF,
                "max_features": (DEFAULT_MAX_FEATURES),
            },
            "structural_feature_count": (6 if self.include_structural_features else 0),
        }
