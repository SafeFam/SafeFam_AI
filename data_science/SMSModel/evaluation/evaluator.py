"""학습, validation threshold 선택, test 평가 흐름"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from data_science.SMSModel.evaluation.latency import (
    LatencyMetrics,
    measure_single_inference_latency,
)
from data_science.SMSModel.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from data_science.SMSModel.evaluation.threshold import (
    ThresholdSelection,
    select_validation_threshold,
)
from data_science.SMSModel.modeling import (
    BasePhishingClassifier,
)


@dataclass(frozen=True)
class ModelEvaluationResult:
    """단일 모델의 validation 선택 및 test 평가 결과"""

    model_name: str
    score_type: str
    selected_threshold: float
    validation: ThresholdSelection
    test_metrics: ClassificationMetrics
    latency: LatencyMetrics
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:

        """평과 결과를 딕셔너리 형태로 직렬화"""
        return {
            "model_name": self.model_name,
            "score_type": self.score_type,
            "selected_threshold": (
                self.selected_threshold
            ),
            "validation": asdict(self.validation),
            "test_metrics": self.test_metrics.to_dict(),
            "latency": asdict(self.latency),
            "metadata": self.metadata,
        }


def train_and_evaluate_model(
    model: BasePhishingClassifier,
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_recall: float = 0.96,
    latency_sample_count: int = 100,
) -> ModelEvaluationResult:

    """단일 모델의 공통 학습 및 평가 흐름을 실행"""

    # 학습 데이터셋으로 모델 학습
    model.fit(train_df)

    # Validation 데이터셋에서 목표 Recall을 달성하는 최적 임계값
    validation_scores = model.predict_scores(
        validation_df
    )

    threshold_selection = select_validation_threshold(
        validation_df["label"].to_numpy(),
        validation_scores.values,
        target_recall=target_recall,
    )

    # validation에서 확정된 threshold를 변경하지 않고 test에 적용
    test_scores = model.predict_scores(test_df)

    if test_scores.score_type != validation_scores.score_type:
        raise ValueError(
            "model score type changed between validation and test"
        )

    # 이미 계산한 test score를 재사용
    test_predictions = model.labels_from_scores(
        test_scores.values,
        threshold=threshold_selection.threshold,
    )

    test_metrics = calculate_classification_metrics(
        test_df["label"].to_numpy(),
        test_predictions,
    )

    # Test 데이터 기반 실시간 단건 추론 지연시간 측정
    latency = measure_single_inference_latency(
        model,
        test_df,
        sample_count=latency_sample_count,
    )

    # 최종 종합 평가 결과 반환
    return ModelEvaluationResult(
        model_name=model.model_name,
        score_type=test_scores.score_type.value,
        selected_threshold=(
            threshold_selection.threshold
        ),
        validation=threshold_selection,
        test_metrics=test_metrics,
        latency=latency,
        metadata=model.get_metadata(),
    )