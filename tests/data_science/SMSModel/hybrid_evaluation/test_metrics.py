"""하이브리드 평가 순수 메트릭 함수 테스트."""

import pytest

from data_science.SMSModel.hybrid_evaluation import (
    EvaluationMode,
    EvaluationRecord,
    OperationalOutcome,
    TokenUsage,
    calculate_classification_metrics,
    calculate_cost_metrics,
    calculate_cost_reduction_rate,
    calculate_full_dataset_metrics,
    calculate_latency_metrics,
    calculate_operational_metrics,
)


def test_evaluation_record_serializes_provider_neutral_metadata() -> None:
    """민감한 원문 없이 Bedrock 재현성 메타데이터를 직렬화합니다."""

    record = EvaluationRecord(
        sample_id="sha256:anonymous",
        mode=EvaluationMode.HYBRID,
        expected_label="phishing",
        predicted_label="phishing",
        latency_ms=123.4,
        result_available=True,
        llm_called=True,
        llm_available=True,
        fallback_applied=False,
        all_engines_unavailable=False,
        decision_source="LLM",
        routing_decision="CALL_LLM",
        routing_reason="UNCERTAIN",
        error_code=None,
        llm_provider="AWS_BEDROCK",
        llm_model="anthropic.claude-haiku-4-5",
        input_tokens=100,
        output_tokens=20,
    )

    payload = record.to_dict()

    assert payload["mode"] == "HYBRID"
    assert payload["result_available"] is True
    assert payload["llm_provider"] == "AWS_BEDROCK"
    assert payload["llm_model"] == "anthropic.claude-haiku-4-5"
    assert "text" not in payload


def test_calculates_classification_metrics() -> None:
    """Accuracy, F-score와 혼동 행렬을 함께 계산합니다."""

    metrics = calculate_classification_metrics(
        y_true=[
            "normal",
            "normal",
            "phishing",
            "phishing",
        ],
        y_pred=[
            "normal",
            "phishing",
            "normal",
            "phishing",
        ],
    )

    assert metrics.sample_count == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.f2 == pytest.approx(0.5)

    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1


def test_classification_metrics_handle_no_positive_prediction() -> None:
    """피싱 예측이 없어도 0으로 나누지 않습니다."""

    metrics = calculate_classification_metrics(
        y_true=[
            "normal",
            "phishing",
        ],
        y_pred=[
            "normal",
            "normal",
        ],
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.f2 == 0.0
    assert metrics.false_negative == 1


def test_full_dataset_metrics_match_binary_metrics_when_all_available() -> None:
    """모든 결과가 있으면 전체 정확도는 이진 분류 정확도와 같습니다."""

    metrics = calculate_full_dataset_metrics(
        y_true=["normal", "normal", "phishing", "phishing"],
        y_pred=["normal", "phishing", "normal", "phishing"],
    )

    assert metrics.total_sample_count == 4
    assert metrics.available_count == 4
    assert metrics.unavailable_count == 0
    assert metrics.correct_count == 2
    assert metrics.incorrect_count == 2
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.actual_normal_count == 2
    assert metrics.actual_phishing_count == 2
    assert metrics.detected_phishing_count == 1
    assert metrics.missed_phishing_count == 1
    assert metrics.phishing_detection_rate == pytest.approx(0.5)


def test_full_dataset_metrics_count_unknown_as_unavailable_failure() -> None:
    """UNKNOWN은 전체 분모와 피싱 미탐에 포함합니다."""

    metrics = calculate_full_dataset_metrics(
        y_true=["normal", "phishing", "phishing"],
        y_pred=["unknown", "unknown", "phishing"],
    )

    assert metrics.total_sample_count == 3
    assert metrics.available_count == 1
    assert metrics.unavailable_count == 2
    assert metrics.correct_count == 1
    assert metrics.incorrect_count == 0
    assert metrics.accuracy == pytest.approx(1 / 3)
    assert metrics.detected_phishing_count == 1
    assert metrics.missed_phishing_count == 1
    assert metrics.phishing_detection_rate == pytest.approx(0.5)
    assert (
        metrics.correct_count
        + metrics.incorrect_count
        + metrics.unavailable_count
        == metrics.total_sample_count
    )


def test_full_dataset_metrics_handle_no_phishing_samples() -> None:
    """피싱 표본이 없으면 탐지율을 0으로 반환합니다."""

    metrics = calculate_full_dataset_metrics(
        y_true=["normal", "normal"],
        y_pred=["normal", "unknown"],
    )

    assert metrics.actual_phishing_count == 0
    assert metrics.detected_phishing_count == 0
    assert metrics.missed_phishing_count == 0
    assert metrics.phishing_detection_rate == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "error_message"),
    [
        ([], [], "empty labels"),
        (["normal"], ["normal", "unknown"], "same length"),
        (["safe"], ["normal"], "y_true contains unsupported labels"),
        (["normal"], ["unavailable"], "y_pred contains unsupported labels"),
    ],
)
def test_rejects_invalid_full_dataset_inputs(
    y_true: list[str],
    y_pred: list[str],
    error_message: str,
) -> None:
    """전체 지표도 빈 입력, 길이 및 label 계약을 검증합니다."""

    with pytest.raises(ValueError, match=error_message):
        calculate_full_dataset_metrics(y_true, y_pred)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "error_message"),
    [
        (
            [],
            [],
            "empty labels",
        ),
        (
            ["normal"],
            ["normal", "phishing"],
            "same length",
        ),
        (
            ["safe"],
            ["normal"],
            "y_true contains unsupported labels",
        ),
        (
            ["normal"],
            ["unknown"],
            "y_pred contains unsupported labels",
        ),
    ],
)
def test_rejects_invalid_classification_inputs(
    y_true: list[str],
    y_pred: list[str],
    error_message: str,
) -> None:
    """빈 입력, 길이 불일치 및 알 수 없는 label을 거부합니다."""

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        calculate_classification_metrics(
            y_true,
            y_pred,
        )


def test_calculates_latency_distribution() -> None:
    """평균과 분위수 latency를 계산합니다."""

    metrics = calculate_latency_metrics(
        [10.0, 20.0, 30.0, 40.0],
    )

    assert metrics.sample_count == 4
    assert metrics.average_ms == pytest.approx(25.0)
    assert metrics.p50_ms == pytest.approx(25.0)
    assert metrics.p95_ms == pytest.approx(38.5)
    assert metrics.minimum_ms == pytest.approx(10.0)
    assert metrics.maximum_ms == pytest.approx(40.0)


@pytest.mark.parametrize(
    "durations_ms",
    [
        [],
        [-1.0],
        [float("nan")],
        [float("inf")],
        [True],
    ],
)
def test_rejects_invalid_latency_values(
    durations_ms,
) -> None:
    """빈 값, 음수, NaN, 무한대 및 bool을 거부합니다."""

    with pytest.raises(ValueError):
        calculate_latency_metrics(
            durations_ms
        )


def test_calculates_operational_metrics() -> None:
    """호출률과 fallback 경로를 집계합니다."""

    outcomes = [
        # 확실한 자체 모델 판정: LLM 미호출
        OperationalOutcome(
            llm_called=False,
            llm_available=False,
            fallback_applied=False,
            all_engines_unavailable=False,
        ),
        # LLM 호출 성공
        OperationalOutcome(
            llm_called=True,
            llm_available=True,
            fallback_applied=False,
            all_engines_unavailable=False,
        ),
        # LLM 실패 후 stacking fallback
        OperationalOutcome(
            llm_called=True,
            llm_available=False,
            fallback_applied=True,
            all_engines_unavailable=False,
        ),
        # 두 엔진 모두 실패
        OperationalOutcome(
            llm_called=True,
            llm_available=False,
            fallback_applied=True,
            all_engines_unavailable=True,
        ),
    ]

    metrics = calculate_operational_metrics(
        outcomes
    )

    assert metrics.total_message_count == 4
    assert metrics.llm_call_count == 3
    assert metrics.llm_call_rate == pytest.approx(0.75)
    assert metrics.llm_success_count == 1
    assert metrics.llm_failure_count == 2
    assert metrics.fallback_count == 2
    assert (
        metrics.all_engines_unavailable_count
        == 1
    )


def test_self_model_mode_has_zero_llm_call_rate() -> None:
    """자체 모델 단독 모드는 LLM 호출률이 0이어야 합니다."""

    metrics = calculate_operational_metrics(
        [
            OperationalOutcome(
                llm_called=False,
                llm_available=False,
                fallback_applied=False,
                all_engines_unavailable=False,
            )
            for _ in range(5)
        ]
    )

    assert metrics.llm_call_count == 0
    assert metrics.llm_call_rate == 0.0
    assert metrics.llm_success_count == 0
    assert metrics.llm_failure_count == 0


def test_rejects_empty_operational_outcomes() -> None:
    """0건으로 호출률을 계산하지 않습니다."""

    with pytest.raises(
        ValueError,
        match="empty outcomes",
    ):
        calculate_operational_metrics([])


def test_calculates_cost_from_bedrock_usage() -> None:
    """측정 가능한 Bedrock token usage로 비용을 계산합니다."""

    metrics = calculate_cost_metrics(
        usages=[
            TokenUsage(
                input_tokens=1_000,
                output_tokens=200,
            ),
            TokenUsage(
                input_tokens=500,
                output_tokens=100,
            ),
            # timeout 등으로 usage를 얻지 못한 호출
            TokenUsage(
                input_tokens=None,
                output_tokens=None,
            ),
        ],
        total_message_count=10,
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    )

    # 입력 비용:
    # 1,500 / 1,000,000 × 1.0 = 0.0015
    #
    # 출력 비용:
    # 300 / 1,000,000 × 5.0 = 0.0015
    #
    # 전체 비용:
    # 0.0015 + 0.0015 = 0.003
    assert metrics.input_tokens == 1_500
    assert metrics.output_tokens == 300
    assert metrics.measured_call_count == 2
    assert metrics.unmeasured_call_count == 1
    assert metrics.total_cost == pytest.approx(
        0.003
    )
    assert metrics.cost_per_message == pytest.approx(
        0.0003
    )


def test_zero_llm_calls_produce_zero_cost() -> None:
    """자체 모델 단독 모드는 token과 비용이 모두 0입니다."""

    metrics = calculate_cost_metrics(
        usages=[],
        total_message_count=10,
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    )

    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.measured_call_count == 0
    assert metrics.unmeasured_call_count == 0
    assert metrics.total_cost == 0.0
    assert metrics.cost_per_message == 0.0


@pytest.mark.parametrize(
    (
        "total_message_count",
        "input_price",
        "output_price",
    ),
    [
        (0, 1.0, 5.0),
        (-1, 1.0, 5.0),
        (10, -1.0, 5.0),
        (10, 1.0, -5.0),
        (10, float("nan"), 5.0),
        (10, 1.0, float("inf")),
    ],
)
def test_rejects_invalid_cost_inputs(
    total_message_count,
    input_price,
    output_price,
) -> None:
    """잘못된 메시지 수와 token 단가를 거부합니다."""

    with pytest.raises(ValueError):
        calculate_cost_metrics(
            usages=[],
            total_message_count=(
                total_message_count
            ),
            input_price_per_million=(
                input_price
            ),
            output_price_per_million=(
                output_price
            ),
        )


def test_incomplete_usage_is_unmeasured() -> None:
    """input/output 중 하나라도 없으면 비용에서 제외합니다."""

    metrics = calculate_cost_metrics(
        usages=[
            TokenUsage(
                input_tokens=100,
                output_tokens=None,
            ),
            TokenUsage(
                input_tokens=None,
                output_tokens=50,
            ),
            TokenUsage(
                input_tokens=-1,
                output_tokens=50,
            ),
        ],
        total_message_count=3,
        input_price_per_million=1.0,
        output_price_per_million=5.0,
    )

    assert metrics.measured_call_count == 0
    assert metrics.unmeasured_call_count == 3
    assert metrics.total_cost == 0.0


def test_calculates_cost_reduction_rate() -> None:
    """LLM-only 대비 75% 비용 절감을 계산합니다."""

    rate = calculate_cost_reduction_rate(
        llm_only_cost_per_message=0.01,
        hybrid_cost_per_message=0.0025,
    )

    assert rate == pytest.approx(0.75)


def test_cost_increase_produces_negative_reduction() -> None:
    """하이브리드 비용이 더 높으면 감소율은 음수입니다."""

    rate = calculate_cost_reduction_rate(
        llm_only_cost_per_message=0.01,
        hybrid_cost_per_message=0.015,
    )

    assert rate == pytest.approx(-0.5)


def test_zero_baseline_cost_has_no_reduction_rate() -> None:
    """기준 비용이 0이면 비용 감소율을 정의할 수 없습니다."""

    rate = calculate_cost_reduction_rate(
        llm_only_cost_per_message=0.0,
        hybrid_cost_per_message=0.0,
    )

    assert rate is None


@pytest.mark.parametrize(
    ("baseline", "hybrid"),
    [
        (-1.0, 0.0),
        (1.0, -1.0),
        (float("nan"), 0.0),
        (1.0, float("inf")),
    ],
)
def test_rejects_invalid_cost_reduction_inputs(
    baseline: float,
    hybrid: float,
) -> None:
    """음수, NaN 및 무한대 비용을 거부합니다."""

    with pytest.raises(ValueError):
        calculate_cost_reduction_rate(
            llm_only_cost_per_message=baseline,
            hybrid_cost_per_message=hybrid,
        )
