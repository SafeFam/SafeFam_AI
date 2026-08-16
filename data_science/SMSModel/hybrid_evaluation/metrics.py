"""하이브리드 평가 결과를 계산하는 순수 함수"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from data_science.SMSModel.hybrid_evaluation.models import (
    ClassificationMetrics,
    CostMetrics,
    FullDatasetMetrics,
    LatencyMetrics,
    OperationalMetrics,
    OperationalOutcome,
    TokenUsage,
)

LABEL_ORDER = ("normal", "phishing")
UNAVAILABLE_LABEL = "unknown"

def _validate_binary_labels(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """분류 메트릭 입력을 검증하고 1차원 배열로 변환"""

    true_values = np.asarray(y_true, dtype=str)
    predicted_values = np.asarray(y_pred, dtype=str)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional"
        )

    if len(true_values) == 0:
        raise ValueError(
            "cannot calculate metrics from empty labels"
        )

    if len(true_values) != len(predicted_values):
        raise ValueError(
            "y_true and y_pred must have the same length"
        )

    allowed_labels = set(LABEL_ORDER)

    if not set(true_values).issubset(allowed_labels):
        raise ValueError(
            "y_true contains unsupported labels"
        )

    if not set(predicted_values).issubset(allowed_labels):
        raise ValueError(
            "y_pred contains unsupported labels"
        )

    return true_values, predicted_values

def calculate_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],   
) -> ClassificationMetrics:
    """Accuracy, Precision, Recall, F1, F2와 혼동 행렬을 계산"""

    true_values, predicted_values = (
        _validate_binary_labels(
            y_true,
            y_pred,
        )
    )

    matrix = confusion_matrix(
        true_values,
        predicted_values,
        labels=LABEL_ORDER,
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    return ClassificationMetrics(
        sample_count=len(true_values),
        accuracy=float(
            accuracy_score(
                true_values,
                predicted_values,
            )
        ),
        precision=float(
            precision_score(
                true_values,
                predicted_values,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                true_values,
                predicted_values,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                true_values,
                predicted_values,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        f2=float(
            fbeta_score(
                true_values,
                predicted_values,
                beta=2,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def calculate_full_dataset_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> FullDatasetMetrics:
    """UNKNOWN을 실패로 포함한 전체 데이터셋 지표를 계산"""

    true_values = np.asarray(y_true, dtype=str)
    predicted_values = np.asarray(y_pred, dtype=str)

    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be one-dimensional"
        )

    if len(true_values) == 0:
        raise ValueError(
            "cannot calculate full-dataset metrics from empty labels"
        )

    if len(true_values) != len(predicted_values):
        raise ValueError(
            "y_true and y_pred must have the same length"
        )

    binary_labels = set(LABEL_ORDER)

    if not set(true_values).issubset(binary_labels):
        raise ValueError(
            "y_true contains unsupported labels"
        )

    allowed_predictions = {*binary_labels, UNAVAILABLE_LABEL}

    if not set(predicted_values).issubset(allowed_predictions):
        raise ValueError(
            "y_pred contains unsupported labels"
        )

    available_mask = np.isin(predicted_values, LABEL_ORDER)
    unavailable_mask = predicted_values == UNAVAILABLE_LABEL
    correct_mask = available_mask & (true_values == predicted_values)
    incorrect_mask = available_mask & (true_values != predicted_values)
    phishing_mask = true_values == "phishing"
    detected_phishing_mask = phishing_mask & (
        predicted_values == "phishing"
    )

    total_sample_count = len(true_values)
    available_count = int(available_mask.sum())
    unavailable_count = int(unavailable_mask.sum())
    correct_count = int(correct_mask.sum())
    incorrect_count = int(incorrect_mask.sum())
    actual_normal_count = int((true_values == "normal").sum())
    actual_phishing_count = int(phishing_mask.sum())
    detected_phishing_count = int(detected_phishing_mask.sum())
    missed_phishing_count = (
        actual_phishing_count - detected_phishing_count
    )

    phishing_detection_rate = (
        detected_phishing_count / actual_phishing_count
        if actual_phishing_count > 0
        else 0.0
    )

    return FullDatasetMetrics(
        total_sample_count=total_sample_count,
        available_count=available_count,
        unavailable_count=unavailable_count,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        accuracy=correct_count / total_sample_count,
        actual_normal_count=actual_normal_count,
        actual_phishing_count=actual_phishing_count,
        detected_phishing_count=detected_phishing_count,
        missed_phishing_count=missed_phishing_count,
        phishing_detection_rate=phishing_detection_rate,
    )

def calculate_latency_metrics(
        durations_ms: Sequence[float],
) -> LatencyMetrics:
    """평균, p50, p95, 최솟값 및 최댓값을 계산"""
    # latency는 모델 내부의 latency가 아니라 각 평가 모드의 end-to-end 처리 시간을 사용
    # 성공과 실패 요청을 모두 포함

    if len(durations_ms) == 0:
        raise ValueError(
            "cannot calculate latency from empty values"
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        for value in durations_ms
    ):
        raise ValueError(
            "latency values must be numeric"
        )

    values = np.asarray(
        durations_ms,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "latency values must be finite"
        )

    if (values < 0).any():
        raise ValueError(
            "latency values must not be negative"
        )

    return LatencyMetrics(
        sample_count=len(values),
        average_ms=float(values.mean()),
        p50_ms=float(
            np.percentile(values, 50)
        ),
        p95_ms=float(
            np.percentile(values, 95)
        ),
        minimum_ms=float(values.min()),
        maximum_ms=float(values.max()),
    )

def calculate_operational_metrics(
        outcomes: Sequence[OperationalOutcome],
) -> OperationalMetrics:
    """LLM 호출률, 성공실패 및 fallback 경로 집계"""

    if len(outcomes) == 0:
        raise ValueError(
            "cannot calculate operational metrics "
            "from empty outcomes"
        )

    if any(not isinstance(outcome, OperationalOutcome) for outcome in outcomes):
        raise TypeError("outcomes must contain OperationalOutcome values")

    # LLM이 실제 호출된 메시지 수
    llm_call_count = sum(
        outcome.llm_called
        for outcome in outcomes
    )

    # 호출됐고 사용 가능한 결과를 반환한 메시지 수
    llm_success_count = sum(
        outcome.llm_called
        and outcome.llm_available
        for outcome in outcomes
    )

    # 호출됐지만 사용할 수 있는 결과가 없었던 메시지 수
    llm_failure_count = sum(
        outcome.llm_called
        and not outcome.llm_available
        for outcome in outcomes
    )

    fallback_count = sum(
        outcome.fallback_applied
        for outcome in outcomes
    )

    all_engines_unavailable_count = sum(
        outcome.all_engines_unavailable
        for outcome in outcomes
    )

    total_message_count = len(outcomes)

    return OperationalMetrics(
        total_message_count=total_message_count,
        llm_call_count=llm_call_count,
        llm_call_rate=(
            llm_call_count
            / total_message_count
        ),
        llm_success_count=llm_success_count,
        llm_failure_count=llm_failure_count,
        fallback_count=fallback_count,
        all_engines_unavailable_count=(
            all_engines_unavailable_count
        ),
    )

def _validate_price(
        value: float,
        *,
        field_name: str,
) -> float:
    """가격 값이 유한한 0 이상의 숫자인지 검증"""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if normalized < 0:
        raise ValueError(
            f"{field_name} must not be negative"
        )

    return normalized

def _is_valid_token_count(
        value: int | None,
) -> bool:
    """비용 계산에 사용할 수 있는 token 값인지 확인"""

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )

def calculate_cost_metrics(
    usages: Sequence[TokenUsage],
    *,
    total_message_count: int,
    input_price_per_million: float,
    output_price_per_million: float,    
) -> CostMetrics:
    """Bedrock token usage와 단가로 비용을 계산"""
    # 단가 단위는 token 1,000,000개당 비용

    if (
        isinstance(total_message_count, bool)
        or not isinstance(total_message_count, int)
        or total_message_count <= 0
    ):
        raise ValueError(
            "total_message_count must be a positive integer"
        )

    input_price = _validate_price(
        input_price_per_million,
        field_name="input_price_per_million",
    )

    output_price = _validate_price(
        output_price_per_million,
        field_name="output_price_per_million",
    )

    total_input_tokens = 0
    total_output_tokens = 0
    measured_call_count = 0
    unmeasured_call_count = 0

    for usage in usages:
        if not isinstance(usage, TokenUsage):
            raise TypeError(
                "usages must contain TokenUsage instances"
            )

        if not (
            _is_valid_token_count(
                usage.input_tokens
            )
            and _is_valid_token_count(
                usage.output_tokens
            )
        ):
            unmeasured_call_count += 1
            continue

        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        measured_call_count += 1

    input_cost = (
        total_input_tokens
        * input_price
        / 1_000_000
    )

    output_cost = (
        total_output_tokens
        * output_price
        / 1_000_000
    )

    total_cost = input_cost + output_cost

    return CostMetrics(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        measured_call_count=measured_call_count,
        unmeasured_call_count=unmeasured_call_count,
        total_cost=total_cost,
        cost_per_message=(
            total_cost
            / total_message_count
        ),
    )

def calculate_cost_reduction_rate(
    *,
    llm_only_cost_per_message: float,
    hybrid_cost_per_message: float,   
) -> float | None:
    """LLM-only 대비 하이브리드 비용 감소율을 계산"""
    # 계산식: 1 - hybrid_cost_per_message / llm_only_cost_per_message

    baseline_cost = _validate_price(
        llm_only_cost_per_message,
        field_name="llm_only_cost_per_message",
    )

    hybrid_cost = _validate_price(
        hybrid_cost_per_message,
        field_name="hybrid_cost_per_message",
    )

    if baseline_cost == 0:
        return None

    return (
        1.0
        - hybrid_cost
        / baseline_cost
    )
