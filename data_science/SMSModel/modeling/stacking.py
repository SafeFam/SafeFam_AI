"""OOF 예측값을 이용하는 SMS 피싱 stacking 분류기"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from app.analysis.text.preprocessing import URL_PATTERN, normalize_text
from app.analysis.text.structural_features import (
    STACKING_STRUCTURAL_FEATURE_NAMES,
    extract_stacking_structural_matrix,
)
from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)
from data_science.SMSModel.modeling.korean_encoder import (
    KoreanEncoderPhishingClassifier,
)
from data_science.SMSModel.modeling.linear_svm import (
    LinearSvmPhishingClassifier,
)
from data_science.SMSModel.modeling.logistic_regression import (
    LogisticRegressionPhishingClassifier,
)
from data_science.SMSModel.modeling.naive_bayes import (
    NaiveBayesPhishingClassifier,
)


DEFAULT_OOF_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


def _build_text_only_naive_bayes() -> BasePhishingClassifier:
    """구조 특징을 중복 사용하지 않는 Naive Bayes를 생성"""

    return NaiveBayesPhishingClassifier(
        include_structural_features=False
    )

@dataclass(frozen=True)
class StackingPrediction:
    """자체 stacking 모델의 단일 메시지 예측 결과"""

    risk_probability: float
    risk_score: int
    confidence: float
    is_suspected_phishing: bool
    threshold: float
    model_scores: dict[str, float]
    unavailable_models: tuple[str, ...]

def _default_base_model_factories(
) -> dict[str, Callable[[], BasePhishingClassifier]]:
    """각 fold마다 새 모델을 만들 수 있는 factory를 반환"""

    return {
        # 구조 특징은 meta-classifier에 별도로 제공
        # 따라서 NB에는 구조 특징을 다시 넣지 않아 중복 반영을 피함
        "naive_bayes": _build_text_only_naive_bayes,
        "logistic_regression": LogisticRegressionPhishingClassifier,
        "linear_svm": LinearSvmPhishingClassifier,
        # 학습 표본이 유형당 2~6건인 구간을 사전학습 표현으로 보완 (#98)
        "korean_encoder": KoreanEncoderPhishingClassifier,
    }

def normalize_base_scores(output: ScoreOutput) -> np.ndarray:
    """모델별 점수를 공통 0~1 범위로 변환"""

    values = np.asarray(output.values, dtype=np.float64)

    if output.score_type == ScoreType.PROBABILITY:
        return np.clip(values, 0.0, 1.0)

    if output.score_type == ScoreType.DECISION:
        # LinearSVC decision score는 범위 제한이 없으므로 sigmoid로 변환
        return expit(values)

    raise ValueError(f"unsupported score type: {output.score_type}")

class StackingPhishingClassifier:
    """세 텍스트 모델과 구조 특징을 결합하는 meta-clasifier"""

    def __init__(
            self,
        *,
        n_splits: int = DEFAULT_OOF_SPLITS,
        random_state: int = DEFAULT_RANDOM_STATE,
        threshold: float = 0.5,
        base_model_factories: (
            dict[str, Callable[[], BasePhishingClassifier]] | None
        ) = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.n_splits = n_splits
        self.random_state = random_state
        self.threshold = threshold
        self.base_model_factories = (
            base_model_factories or _default_base_model_factories()
        )

        self.meta_classifier = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1_000,
            solver="liblinear",
        )

        self.base_models: dict[str, BasePhishingClassifier] = {}
        self.meta_feature_names: tuple[str, ...] | None = None
        self.oof_probabilities_: np.ndarray | None = None
        self.oof_labels_: np.ndarray | None = None
        self._is_fitted = False

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        *,
        require_label: bool,
        require_group: bool,
    ) -> None:
        required = {"text", "text_norm", "has_url"}

        if require_label:
            required.add("label")

        if require_group:
            required.add("template_group_id")

        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"missing stacking columns: {missing}")

        if df.empty:
            raise ValueError("stacking input DataFrame is empty")

        if df[list(required)].isna().any().any():
            raise ValueError("stacking input contains missing values")

        if require_label:
            labels = set(df["label"].astype(str))

            if labels != {"normal", "phishing"}:
                raise ValueError(
                    "training data must contain normal and phishing labels"
                )

    @property
    def model_names(self) -> tuple[str, ...]:
        """meta feature에 사용되는 모델 순서를 고정"""

        return tuple(self.base_model_factories)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """meta-classifier 입력 열 이름을 반환"""

        return (
            *(f"{name}_score" for name in self.model_names),
            *STACKING_STRUCTURAL_FEATURE_NAMES,
        )

    def _build_meta_features(
        self,
        *,
        base_scores: dict[str, np.ndarray],
        structural_features: np.ndarray,
    ) -> np.ndarray:
        """모델 점수와 구조 특징을 하나의 행렬로 결합"""

        score_columns = [
            np.asarray(base_scores[name], dtype=np.float64)
            for name in self.model_names
        ]
        return np.column_stack(
            [
                *score_columns,
                structural_features,
            ]
        )

    def _generate_oof_features(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """train split 내부에서 OOF 예측값 생성

        나중에 메타 레벨 OOF(임계값 보정용)를 같은 fold 분할로 다시
        재현할 수 있도록 fold별 (train_indices, oof_indices)도 함께 반환한다.
        """

        labels = train_df["label"].astype(str).to_numpy()
        groups = train_df["template_group_id"].astype(str).to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        base_scores = {
            name: np.full(len(train_df), np.nan, dtype=np.float64)
            for name in self.model_names
        }
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for train_indices, oof_indices in splitter.split(
            train_df,
            y=labels,
            groups=groups,
        ):
            folds.append((train_indices, oof_indices))
            fold_train = train_df.iloc[train_indices].reset_index(drop=True)
            fold_oof = train_df.iloc[oof_indices].reset_index(drop=True)

            for name, factory in self.base_model_factories.items():
                model = factory()
                model.fit(fold_train)
                output = model.predict_scores(fold_oof)

                base_scores[name][oof_indices] = normalize_base_scores(output)

        for name, scores in base_scores.items():
            if np.isnan(scores).any():
                raise RuntimeError(
                    f"OOF predictions are incomplete for model: {name}"
                )

        structural_features = extract_stacking_structural_matrix(
            train_df["text"].astype(str)
        )

        oof_features = self._build_meta_features(
            base_scores=base_scores,
            structural_features=structural_features,
        )
        return oof_features, folds

    def _generate_meta_oof_probabilities(
        self,
        oof_features: np.ndarray,
        labels: pd.Series,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """메타 분류기 자체도 같은 fold로 out-of-fold 확률을 낸다.

        base 모델의 OOF 피처(oof_features)는 자기 행을 본 적 없는 모델이
        만든 값이지만, 메타 분류기를 그 피처+라벨 전체로 학습한 뒤 같은
        데이터에 predict_proba를 돌리면 메타 분류기 자신에게는 여전히
        재대입(resubstitution)이라 낙관적으로 잡힌다. 같은 fold 분할을
        재사용해 fold별로 메타 분류기를 다시 학습·예측하면, 각 fold의
        확률이 그 fold를 전혀 보지 않은 메타 분류기에서 나온다 - 임계값을
        internal validation(수십~백여 건) 대신 train pool 전체 규모로
        보정할 때 이 정직한 확률이 필요하다 (그렇지 않으면 재대입
        낙관 편향이 임계값을 그대로 왜곡한다).
        """
        phishing_probabilities = np.full(len(labels), np.nan, dtype=np.float64)

        for train_indices, oof_indices in folds:
            fold_meta = LogisticRegression(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1_000,
                solver="liblinear",
            )
            fold_meta.fit(
                oof_features[train_indices],
                labels.iloc[train_indices],
            )
            phishing_index = list(fold_meta.classes_).index("phishing")
            phishing_probabilities[oof_indices] = fold_meta.predict_proba(
                oof_features[oof_indices]
            )[:, phishing_index]

        if np.isnan(phishing_probabilities).any():
            raise RuntimeError("meta OOF probabilities are incomplete")

        return phishing_probabilities

    def fit(self, train_df: pd.DataFrame) -> Self:
        """OOF 예측으로 meta 모델을 학습하고 base 모델을 재학습"""

        self._validate_dataframe(
            train_df,
            require_label=True,
            require_group=True,
        )

        oof_features, folds = self._generate_oof_features(train_df)
        labels = train_df["label"].astype(str)

        # meta-classifier는 base 모델이 자기 학습 행을 보지 않은 OOF 예측값만 사용해 학습
        self.meta_classifier.fit(oof_features, labels)

        # 임계값/구간 보정용 - 메타 분류기까지 out-of-fold인 정직한 확률.
        # internal validation(수십~백여 건)보다 train pool 전체 규모라
        # 임계값 선정의 표본 분산이 훨씬 작다.
        self.oof_probabilities_ = self._generate_meta_oof_probabilities(
            oof_features, labels, folds
        )
        self.oof_labels_ = labels.to_numpy()

        # 운영 추론해 사용할 base 모델은 전체 train split으로 다시 학습
        self.base_models = {}

        for name, factory in self.base_model_factories.items():
            model = factory()
            model.fit(train_df)
            self.base_models[name] = model

        self.meta_feature_names = self.feature_names
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("stacking classifier is not fitted")

        if set(self.base_models) != set(self.model_names):
            raise RuntimeError("stacking base models are incomplete")

        if not hasattr(self.meta_classifier, "classes_"):
            raise RuntimeError("stacking meta-classifier is unavailable")

    def _predict_base_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[str, ...],
    ]:
        """개별 모델 오류를 격리하며 meta 입력 생성"""

        row_count = len(df)
        base_scores: dict[str, np.ndarray] = {}
        unavailable_models: list[str] = []

        for name in self.model_names:
            try:
                output = self.base_models[name].predict_scores(df)
                base_scores[name] = normalize_base_scores(output)
            except Exception:  # noqa: BLE001 - 모델별 실패를 격리
                # 실패 모델을 0점으로 처리하면 위험 메시지를 정상으로 낮출 수 있으므로 중립값 0.5를 사용
                base_scores[name] = np.full(
                    row_count,
                    0.5,
                    dtype=np.float64,
                )
                unavailable_models.append(name)

        return (
            base_scores,
            tuple(unavailable_models),
        )

    def _predict_probability_details(
        self,
        df: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        dict[str, np.ndarray],
    ]:
        """한 번의 base 추론으로 확률과 단계별 세부 정보를 반환합니다."""

        self._require_fitted()
        self._validate_dataframe(
            df,
            require_label=False,
            require_group=False,
        )

        base_scores, unavailable = (
            self._predict_base_features(df)
        )
        structural_features = extract_stacking_structural_matrix(
            df["text"].astype(str)
        )

        meta_features = self._build_meta_features(
            base_scores=base_scores,
            structural_features=structural_features,
        )

        phishing_index = list(
            self.meta_classifier.classes_
        ).index("phishing")

        probabilities = self.meta_classifier.predict_proba(
            meta_features
        )[:, phishing_index]

        # 모델이 하나라도 실패했을 때 결과가 과도하게 안전 방향으로
        # 이동하지 않도록 사용 가능한 base score의 최댓값을 하한으로 둠
        if unavailable:
            available_scores = [
                base_scores[name]
                for name in self.model_names
                if name not in unavailable
            ]

            if available_scores:
                conservative_floor = np.max(
                    np.column_stack(available_scores),
                    axis=1,
                )
                probabilities = np.maximum(
                    probabilities,
                    conservative_floor,
                )
            else:
                # 모든 모델이 실패한 경우 정상 판정 반환 X
                probabilities = np.ones(len(df), dtype=np.float64)

        return (
            np.clip(probabilities, 0.0, 1.0),
            unavailable,
            base_scores,
        )

    def predict_probabilities(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """피싱 클래스 확률과 사용할 수 없었던 모델 목록을 반환합니다."""

        probabilities, unavailable, _base_scores = (
            self._predict_probability_details(df)
        )
        return probabilities, unavailable

    def predict_one(self, text: str) -> StackingPrediction:
        """단일 메시지의 위험 점수와 신뢰도를 반환"""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # 기존 모델들이 요구하는 공통 DataFrame 계약을 구성
        df = pd.DataFrame(
            [
                {
                    "text": text,
                    "text_norm": normalize_text(text),
                    "has_url": bool(URL_PATTERN.search(text)),
                }
            ]
        )

        probabilities, unavailable, base_scores = (
            self._predict_probability_details(df)
        )
        probability = float(probabilities[0])

        # 실제 판정 임계값에서 멀수록 확신이 높은 것으로 정의
        if probability >= self.threshold:
            confidence = (
                1.0
                if self.threshold == 1.0
                else (probability - self.threshold) / (1.0 - self.threshold)
            )
        else:
            confidence = (
                1.0
                if self.threshold == 0.0
                else (self.threshold - probability) / self.threshold
            )
        confidence = min(1.0, max(0.0, confidence))

        return StackingPrediction(
            risk_probability=probability,
            risk_score=round(probability * 100),
            confidence=confidence,
            is_suspected_phishing=probability >= self.threshold,
            threshold=self.threshold,
            model_scores={
                name: float(values[0])
                for name, values in base_scores.items()
            },
            unavailable_models=unavailable,
        )

    def set_threshold(self, threshold: float) -> None:
        """validation 데이터에서 선정한 최종 임계값을 설정"""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.threshold = float(threshold)

    def get_metadata(self) -> dict[str, Any]:
        """artifact와 보고서에 저장할 비민감 메타데이터"""

        self._require_fitted()

        return {
            "model_name": "stacking_phishing_classifier",
            "base_models": list(self.model_names),
            "meta_classifier": "LogisticRegression",
            "meta_feature_names": list(self.feature_names),
            "oof_splits": self.n_splits,
            "random_state": self.random_state,
            "threshold": self.threshold,
        }
