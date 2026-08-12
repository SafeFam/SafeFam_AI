"""세 가지 분석 모드의 성능·운영·비용 비교 보고서 생성."""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from .metrics import (
    calculate_classification_metrics,
    calculate_cost_metrics,
    calculate_latency_metrics,
    calculate_operational_metrics,
)
from .models import EvaluationMode, OperationalOutcome, TokenUsage

REPORT_SCHEMA_VERSION = 1
_MODES = tuple(EvaluationMode)
_BINARY_LABELS = {"normal", "phishing"}


def build_comparison_report(
    records: Sequence[dict[str, Any]],
    *,
    source_metadata: dict[str, Any],
    policy: dict[str, Any],
    input_price_per_million: float,
    output_price_per_million: float,
    currency: str,
    pricing_as_of: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """검증된 평가 레코드로 JSON 직렬화 가능한 보고서를 생성합니다."""
    grouped, sample_ids = _validate_and_group(records)
    currency = _non_empty_string(currency, "currency")
    pricing_as_of = _non_empty_string(pricing_as_of, "pricing_as_of")
    selection = _validate_policy(policy)
    metadata = _validate_source_metadata(source_metadata)

    mode_reports = {
        mode.value: _summarize_mode(
            grouped[mode],
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
        for mode in _MODES
    }

    self_report = mode_reports[EvaluationMode.SELF_MODEL_ONLY.value]
    llm_report = mode_reports[EvaluationMode.LLM_ONLY.value]
    hybrid_report = mode_reports[EvaluationMode.HYBRID.value]

    comparisons = {
        "llm_call_reduction_rate": _reduction_rate(
            llm_report["operations"]["llm_call_rate"],
            hybrid_report["operations"]["llm_call_rate"],
        ),
        "cost_per_message_reduction_rate": _reduction_rate(
            llm_report["cost"]["cost_per_message"],
            hybrid_report["cost"]["cost_per_message"],
        ),
        "average_latency_reduction_rate": _reduction_rate(
            llm_report["latency_ms"]["average_ms"],
            hybrid_report["latency_ms"]["average_ms"],
        ),
        "p95_latency_reduction_rate": _reduction_rate(
            llm_report["latency_ms"]["p95_ms"],
            hybrid_report["latency_ms"]["p95_ms"],
        ),
        "hybrid_recall_delta_vs_self_model": _metric_delta(
            hybrid_report["classification"],
            self_report["classification"],
            "recall",
        ),
        "hybrid_f2_delta_vs_self_model": _metric_delta(
            hybrid_report["classification"],
            self_report["classification"],
            "f2",
        ),
    }

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "source_split": "test",
        "sample_count": len(sample_ids),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "evaluation_schema_version": metadata["evaluation_schema_version"],
        "model": {
            "provider": "AWS_BEDROCK",
            "model_id": metadata["model_id"],
            "region": metadata["region"],
            "prompt_version": metadata["prompt_version"],
        },
        "pricing": {
            "currency": currency,
            "pricing_as_of": pricing_as_of,
            "unit": "per_1_million_tokens",
            "input_price": float(input_price_per_million),
            "output_price": float(output_price_per_million),
        },
        "routing_policy": {
            "source_split": "validation",
            "normal_probability_max": selection["normal_probability_max"],
            "phishing_probability_min": selection["phishing_probability_min"],
            "validation_llm_call_rate": selection["llm_call_rate"],
            "test_uncertain_count": hybrid_report["operations"]["llm_call_count"],
            "test_uncertain_rate": hybrid_report["operations"]["llm_call_rate"],
        },
        "modes": mode_reports,
        "comparisons": comparisons,
        "classification_policy": {
            "positive_label": "phishing",
            "unknown_handling": (
                "UNKNOWN results are excluded from binary classification "
                "metrics and reported as unavailable_count."
            ),
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """비교 보고서를 사람이 검토할 Markdown으로 변환합니다."""
    modes = report["modes"]
    lines = [
        "# Stacking·Claude·Hybrid 평가 보고서",
        "",
        "## 평가 조건",
        "",
        f"- 생성 시각: `{report['generated_at']}`",
        f"- 평가 split: `{report['source_split']}`",
        f"- 샘플 수: `{report['sample_count']}`",
        f"- 데이터셋 fingerprint: `{report['dataset_fingerprint']}`",
        f"- LLM: `{report['model']['provider']}` / `{report['model']['model_id']}`",
        f"- Region: `{report['model']['region']}`",
        f"- 프롬프트 버전: `{report['model']['prompt_version']}`",
        "",
        "## 라우팅 정책",
        "",
        f"- 정책 선정 split: `{report['routing_policy']['source_split']}`",
        "- 정상 확신 상한: "
        f"`{report['routing_policy']['normal_probability_max']:.8f}`",
        "- 피싱 확신 하한: "
        f"`{report['routing_policy']['phishing_probability_min']:.8f}`",
        "- Validation 예상 LLM 호출률: "
        f"`{_percent(report['routing_policy']['validation_llm_call_rate'])}`",
        "- Test 실제 불확실 구간: "
        f"`{report['routing_policy']['test_uncertain_count']}/"
        f"{report['sample_count']}` "
        f"(`{_percent(report['routing_policy']['test_uncertain_rate'])}`)",
        "",
        "## 모드별 비교",
        "",
        "| 지표 | Stacking only | Claude only | Hybrid |",
        "|---|---:|---:|---:|",
    ]

    labels = {
        EvaluationMode.SELF_MODEL_ONLY.value: "Stacking only",
        EvaluationMode.LLM_ONLY.value: "Claude only",
        EvaluationMode.HYBRID.value: "Hybrid",
    }
    ordered = [modes[mode.value] for mode in _MODES]

    def add_row(label: str, path: tuple[str, str], formatter) -> None:
        values = [mode[path[0]][path[1]] for mode in ordered]
        lines.append(
            f"| {label} | " + " | ".join(formatter(value) for value in values) + " |"
        )

    for metric_name, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("f2", "F2"),
    ):
        add_row(label, ("classification", metric_name), _format_ratio)

    add_row("평균 지연시간 (ms)", ("latency_ms", "average_ms"), _format_number)
    add_row("P50 지연시간 (ms)", ("latency_ms", "p50_ms"), _format_number)
    add_row("P95 지연시간 (ms)", ("latency_ms", "p95_ms"), _format_number)
    add_row("LLM 호출률", ("operations", "llm_call_rate"), _percent)
    add_row("입력 token", ("cost", "input_tokens"), str)
    add_row("출력 token", ("cost", "output_tokens"), str)
    add_row("메시지당 비용", ("cost", "cost_per_message"), _format_cost)
    add_row("미측정 LLM 호출", ("cost", "unmeasured_call_count"), str)
    add_row("Fallback", ("operations", "fallback_count"), str)
    add_row(
        "전체 엔진 실패",
        ("operations", "all_engines_unavailable_count"),
        str,
    )
    add_row("결과 없음", ("availability", "unavailable_count"), str)

    comparisons = report["comparisons"]
    lines.extend(
        [
            "",
            "## Hybrid 개선 효과",
            "",
            f"- Claude-only 대비 LLM 호출 감소율: `{_optional_percent(comparisons['llm_call_reduction_rate'])}`",
            f"- Claude-only 대비 메시지당 비용 감소율: `{_optional_percent(comparisons['cost_per_message_reduction_rate'])}`",
            f"- Claude-only 대비 평균 지연시간 감소율: `{_optional_percent(comparisons['average_latency_reduction_rate'])}`",
            f"- Claude-only 대비 P95 지연시간 감소율: `{_optional_percent(comparisons['p95_latency_reduction_rate'])}`",
            f"- Stacking-only 대비 Recall 변화: `{_optional_delta(comparisons['hybrid_recall_delta_vs_self_model'])}`",
            f"- Stacking-only 대비 F2 변화: `{_optional_delta(comparisons['hybrid_f2_delta_vs_self_model'])}`",
            "",
            "## 비용 가정",
            "",
            f"- 통화: `{report['pricing']['currency']}`",
            f"- 가격 기준일: `{report['pricing']['pricing_as_of']}`",
            "- 입력 token 100만 개당 가격: "
            f"`{report['pricing']['input_price']}`",
            "- 출력 token 100만 개당 가격: "
            f"`{report['pricing']['output_price']}`",
            "",
            "## 판정 정책",
            "",
            "`UNKNOWN` 결과는 정상으로 간주하지 않습니다. 이진 분류 지표에서 제외하고 "
            "각 모드의 `결과 없음` 건수로 별도 기록합니다.",
            "",
        ]
    )

    # labels가 코드/보고서 모드 대응을 문서에 남기도록 사용합니다.
    assert len(labels) == len(ordered)
    return "\n".join(lines)


def _validate_and_group(
    records: Sequence[dict[str, Any]],
) -> tuple[dict[EvaluationMode, list[dict[str, Any]]], set[str]]:
    if not records:
        raise ValueError("evaluation records must not be empty")

    grouped = {mode: [] for mode in _MODES}
    seen: set[tuple[EvaluationMode, str]] = set()
    labels_by_sample: dict[str, str] = {}

    for record in records:
        if not isinstance(record, dict):
            raise TypeError("evaluation records must be objects")
        try:
            mode = EvaluationMode(record.get("mode"))
        except (TypeError, ValueError) as exception:
            raise ValueError("evaluation record mode is invalid") from exception
        sample_id = _non_empty_string(record.get("sample_id"), "sample_id")
        expected_label = record.get("expected_label")
        if expected_label not in _BINARY_LABELS:
            raise ValueError("expected_label must be normal or phishing")
        key = (mode, sample_id)
        if key in seen:
            raise ValueError("duplicate mode/sample evaluation record")
        seen.add(key)
        previous_label = labels_by_sample.setdefault(sample_id, expected_label)
        if previous_label != expected_label:
            raise ValueError("expected label differs across evaluation modes")
        _validate_record_fields(record)
        grouped[mode].append(record)

    expected_ids = set(labels_by_sample)
    for mode, mode_records in grouped.items():
        ids = {str(record["sample_id"]) for record in mode_records}
        if ids != expected_ids:
            raise ValueError(f"sample IDs differ for mode {mode.value}")
        mode_records.sort(key=lambda record: str(record["sample_id"]))
    return grouped, expected_ids


def _validate_record_fields(record: dict[str, Any]) -> None:
    if record.get("predicted_label") not in {*_BINARY_LABELS, "unknown"}:
        raise ValueError("predicted_label is invalid")
    for field in (
        "result_available",
        "llm_called",
        "llm_available",
        "fallback_applied",
        "all_engines_unavailable",
    ):
        if not isinstance(record.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    latency = record.get("latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or latency < 0
    ):
        raise ValueError("latency_ms must be a finite non-negative number")


def _summarize_mode(
    records: Sequence[dict[str, Any]],
    *,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    available = [
        record
        for record in records
        if record["result_available"] is True
        and record["predicted_label"] in _BINARY_LABELS
    ]
    classification = (
        calculate_classification_metrics(
            [str(record["expected_label"]) for record in available],
            [str(record["predicted_label"]) for record in available],
        ).to_dict()
        if available
        else None
    )
    operations = calculate_operational_metrics(
        [
            OperationalOutcome(
                llm_called=record["llm_called"],
                llm_available=record["llm_available"],
                fallback_applied=record["fallback_applied"],
                all_engines_unavailable=record["all_engines_unavailable"],
            )
            for record in records
        ]
    ).to_dict()
    usages = [
        TokenUsage(
            input_tokens=record.get("input_tokens"),
            output_tokens=record.get("output_tokens"),
        )
        for record in records
        if record["llm_called"] is True
    ]
    cost = calculate_cost_metrics(
        usages,
        total_message_count=len(records),
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    ).to_dict()
    return {
        "classification": classification,
        "availability": {
            "available_count": len(available),
            "unavailable_count": len(records) - len(available),
            "availability_rate": len(available) / len(records),
        },
        "latency_ms": calculate_latency_metrics(
            [float(record["latency_ms"]) for record in records]
        ).to_dict(),
        "operations": operations,
        "cost": cost,
    }


def _validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("source_split") != "validation":
        raise ValueError("routing policy must come from validation split")
    selection = policy.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("routing policy selection is missing")
    for field in (
        "normal_probability_max",
        "phishing_probability_min",
        "llm_call_rate",
    ):
        value = selection.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"routing policy {field} is invalid")
    if selection["normal_probability_max"] >= selection["phishing_probability_min"]:
        raise ValueError("routing policy thresholds overlap")
    return selection


def _validate_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("source_split") != "test":
        raise ValueError("evaluation records must come from test split")
    for field in (
        "dataset_fingerprint",
        "model_id",
        "region",
        "prompt_version",
    ):
        _non_empty_string(metadata.get(field), field)
    version = metadata.get("evaluation_schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise ValueError("evaluation_schema_version is invalid")
    return metadata


def _reduction_rate(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return 1.0 - candidate / baseline


def _metric_delta(
    candidate: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    metric: str,
) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate[metric]) - float(baseline[metric])


def _non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _format_cost(value: float) -> str:
    return f"{value:.8f}"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _optional_percent(value: float | None) -> str:
    return "N/A" if value is None else _percent(value)


def _optional_delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.4f}"
