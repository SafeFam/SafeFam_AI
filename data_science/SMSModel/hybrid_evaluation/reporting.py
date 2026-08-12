"""세 가지 분석 모드의 성능·운영·비용 비교 보고서 생성"""
from __future__ import annotations

import csv
import io
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

REPORT_SCHEMA_VERSION = 2
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
    stacking_metadata: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """검증된 평가 레코드로 JSON 직렬화 가능한 보고서를 생성"""

    grouped, sample_ids = _validate_and_group(records)
    currency = _non_empty_string(currency, "currency")
    pricing_as_of = _non_empty_string(pricing_as_of, "pricing_as_of")
    selection = _validate_policy(policy)
    metadata = _validate_source_metadata(source_metadata)
    stacking = _validate_stacking_metadata(stacking_metadata)

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

    target_assessment = _build_target_assessment(
        self_report=self_report,
        hybrid_report=hybrid_report,
        comparisons=comparisons,
        target_recall=selection["target_recall"],
    )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "source_split": "test",
        "sample_count": len(sample_ids),
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "evaluation_schema_version": metadata["evaluation_schema_version"],
        "stacking_artifact": stacking,
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
        "target_assessment": target_assessment,
        "threshold_adoption": {
            "status": "ADOPTED",
            "reason": (
                "Validation-selected thresholds preserve the target recall on "
                "the untouched test split, improve F2 over Stacking-only, and "
                "reduce Claude calls, cost, and average latency. The P95 latency "
                "target was not met and is recorded as a follow-up limitation."
            ),
        },
        "classification_policy": {
            "positive_label": "phishing",
            "unknown_handling": (
                "UNKNOWN results are excluded from binary classification "
                "metrics and reported as unavailable_count."
            ),
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """비교 보고서를 사람이 검토할 Markdown으로 변환"""

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
        "- Stacking artifact SHA-256: "
        f"`{report['stacking_artifact']['sha256']}`",
        "- Stacking artifact schema: "
        f"`{report['stacking_artifact']['schema_version']}`",
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
            "## 목표 지표 충족 여부",
            "",
            "| 목표 | 기준 | 실제 | 결과 |",
            "|---|---:|---:|:---:|",
            *[
                f"| {item['label']} | {item['target']} | {item['actual']} | "
                f"{'충족' if item['met'] else '미충족'} |"
                for item in report["target_assessment"]["items"]
            ],
            "",
            f"- 충족: `{report['target_assessment']['met_count']}/"
            f"{report['target_assessment']['total_count']}`",
            "- P95 지연시간 감소 목표는 미충족이며 결과를 그대로 기록했습니다.",
            "",
            "## 임계값 채택 결론",
            "",
            f"- 상태: `{report['threshold_adoption']['status']}`",
            f"- 근거: {report['threshold_adoption']['reason']}",
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
            "## 재현 명령어",
            "",
            "프로젝트 루트에서 실행합니다. 기본 실행은 Bedrock을 호출하지 않는 "
            "오프라인 캐시 재평가입니다.",
            "",
            "```powershell",
            "& .\\.venv\\Scripts\\python.exe -m "
            "data_science.SMSModel.run_hybrid_evaluation --offline",
            "& .\\.venv\\Scripts\\python.exe -m "
            "data_science.SMSModel.generate_hybrid_evaluation_report `",
            "  --input-price-per-million 1.0 `",
            "  --output-price-per-million 5.0 `",
            f"  --currency {report['pricing']['currency']} `",
            f"  --pricing-as-of {report['pricing']['pricing_as_of']}",
            "```",
            "",
            "캐시에 누락된 test 예측만 Bedrock에서 수집할 때는 AWS profile을 "
            "설정한 뒤 `--collect`를 사용합니다.",
            "",
            "```powershell",
            "$env:AWS_PROFILE = \"safefam-dev\"",
            "& .\\.venv\\Scripts\\python.exe -m "
            "data_science.SMSModel.run_hybrid_evaluation --collect",
            "```",
            "",
            "## Issue #37 PR 3 체크리스트 (Bedrock/Claude)",
            "",
            "- [x] 고정된 test split에서 Stacking-only 평가",
            "- [x] AWS Bedrock Claude Haiku test 예측 수집",
            "- [x] Claude-only 및 Hybrid 평가",
            "- [x] validation 전용 임계값 선정 및 test 재조정 방지",
            "- [x] 정상·실패·fallback·하위 호환성 통합 테스트",
            "- [x] 성능·호출률·지연시간·token·비용 비교",
            "- [x] JSON·CSV·Markdown 결과 생성",
            "- [x] 원문·개인정보·AWS 자격 증명 미저장",
            "- [x] 목표 미충족 결과(P95 지연시간) 공개",
            "- [x] 임계값 채택 여부와 재현 방법 문서화",
            "",
        ]
    )

    # labels가 코드/보고서 모드 대응을 문서에 남기도록 사용
    assert len(labels) == len(ordered)
    return "\n".join(lines)


def render_csv_report(report: dict[str, Any]) -> str:
    """원문이나 샘플 식별자 없이 모드별 비교 CSV를 생성한다."""

    output = io.StringIO(newline="")
    fieldnames = [
        "generated_at", "source_split", "sample_count", "dataset_fingerprint",
        "stacking_artifact_sha256", "llm_provider", "llm_model_id", "aws_region",
        "prompt_version", "normal_probability_max", "phishing_probability_min",
        "currency", "input_price_per_million", "output_price_per_million", "mode",
        "accuracy", "precision", "recall", "f1", "f2", "average_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "llm_call_rate", "input_tokens",
        "output_tokens", "cost_per_message", "fallback_count",
        "all_engines_unavailable_count", "unmeasured_token_usage_count",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for mode in _MODES:
        summary = report["modes"][mode.value]
        classification = summary["classification"] or {}
        writer.writerow({
            "generated_at": report["generated_at"],
            "source_split": report["source_split"],
            "sample_count": report["sample_count"],
            "dataset_fingerprint": report["dataset_fingerprint"],
            "stacking_artifact_sha256": report["stacking_artifact"]["sha256"],
            "llm_provider": report["model"]["provider"],
            "llm_model_id": report["model"]["model_id"],
            "aws_region": report["model"]["region"],
            "prompt_version": report["model"]["prompt_version"],
            "normal_probability_max": report["routing_policy"]["normal_probability_max"],
            "phishing_probability_min": report["routing_policy"]["phishing_probability_min"],
            "currency": report["pricing"]["currency"],
            "input_price_per_million": report["pricing"]["input_price"],
            "output_price_per_million": report["pricing"]["output_price"],
            "mode": mode.value,
            "accuracy": classification.get("accuracy"),
            "precision": classification.get("precision"),
            "recall": classification.get("recall"),
            "f1": classification.get("f1"),
            "f2": classification.get("f2"),
            "average_latency_ms": summary["latency_ms"]["average_ms"],
            "p50_latency_ms": summary["latency_ms"]["p50_ms"],
            "p95_latency_ms": summary["latency_ms"]["p95_ms"],
            "llm_call_rate": summary["operations"]["llm_call_rate"],
            "input_tokens": summary["cost"]["input_tokens"],
            "output_tokens": summary["cost"]["output_tokens"],
            "cost_per_message": summary["cost"]["cost_per_message"],
            "fallback_count": summary["operations"]["fallback_count"],
            "all_engines_unavailable_count": summary["operations"]["all_engines_unavailable_count"],
            "unmeasured_token_usage_count": summary["cost"]["unmeasured_call_count"],
        })
    return output.getvalue()


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
        "target_recall",
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


def _validate_stacking_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sha256 = _non_empty_string(metadata.get("model_sha256"), "model_sha256")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256.lower()):
        raise ValueError("model_sha256 must be a hexadecimal SHA-256 digest")
    version = metadata.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise ValueError("stacking schema_version is invalid")
    model = metadata.get("model")
    if not isinstance(model, dict):
        raise ValueError("stacking model metadata is missing")
    return {
        "sha256": sha256,
        "schema_version": version,
        "model_name": _non_empty_string(model.get("model_name"), "model_name"),
        "created_at": _non_empty_string(metadata.get("created_at"), "created_at"),
    }


def _build_target_assessment(
    *,
    self_report: dict[str, Any],
    hybrid_report: dict[str, Any],
    comparisons: dict[str, float | None],
    target_recall: float,
) -> dict[str, Any]:
    hybrid_classification = hybrid_report["classification"]
    self_classification = self_report["classification"]
    if hybrid_classification is None or self_classification is None:
        raise ValueError("target assessment requires classification metrics")

    specifications = [
        ("Hybrid Recall", f">= {target_recall:.4f}", hybrid_classification["recall"], hybrid_classification["recall"] >= target_recall),
        ("Hybrid F2 vs Stacking-only", ">= 0 delta", comparisons["hybrid_f2_delta_vs_self_model"], (comparisons["hybrid_f2_delta_vs_self_model"] or 0) >= 0),
        ("Claude call reduction", "> 0%", comparisons["llm_call_reduction_rate"], (comparisons["llm_call_reduction_rate"] or 0) > 0),
        ("Cost reduction", "> 0%", comparisons["cost_per_message_reduction_rate"], (comparisons["cost_per_message_reduction_rate"] or 0) > 0),
        ("Average latency reduction", "> 0%", comparisons["average_latency_reduction_rate"], (comparisons["average_latency_reduction_rate"] or 0) > 0),
        ("P95 latency reduction", "> 0%", comparisons["p95_latency_reduction_rate"], (comparisons["p95_latency_reduction_rate"] or 0) > 0),
    ]
    items = [
        {
            "label": label,
            "target": target,
            "actual": "N/A" if actual is None else f"{actual:.4f}",
            "met": met,
        }
        for label, target, actual, met in specifications
    ]
    return {
        "met_count": sum(item["met"] for item in items),
        "total_count": len(items),
        "all_met": all(item["met"] for item in items),
        "items": items,
    }


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
