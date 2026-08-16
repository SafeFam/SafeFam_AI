"""세 모드 비교 보고서 집계 및 렌더링 테스트."""
from __future__ import annotations

from copy import deepcopy

import pytest

from data_science.SMSModel.hybrid_evaluation import (
    build_comparison_report,
    render_csv_report,
    render_markdown_report,
)


def _record(
    sample_id: str,
    mode: str,
    expected: str,
    predicted: str,
    *,
    latency_ms: float,
    llm_called: bool,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    available = predicted != "unknown"
    return {
        "sample_id": sample_id,
        "mode": mode,
        "expected_label": expected,
        "predicted_label": predicted,
        "latency_ms": latency_ms,
        "result_available": available,
        "llm_called": llm_called,
        "llm_available": llm_called and available,
        "fallback_applied": False,
        "all_engines_unavailable": not available,
        "decision_source": "LLM" if llm_called else "STACKING",
        "routing_decision": "LLM_REVIEW" if mode == "HYBRID" and llm_called else None,
        "routing_reason": (
            "UNCERTAIN_SELF_MODEL_PREDICTION"
            if mode == "HYBRID" and llm_called
            else None
        ),
        "error_code": None if available else "ALL_TEXT_ENGINES_UNAVAILABLE",
        "llm_provider": "AWS_BEDROCK" if llm_called else None,
        "llm_model": "test-model" if llm_called else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _records() -> list[dict]:
    return [
        _record("a", "SELF_MODEL_ONLY", "normal", "normal", latency_ms=10, llm_called=False),
        _record("b", "SELF_MODEL_ONLY", "phishing", "normal", latency_ms=20, llm_called=False),
        _record("a", "LLM_ONLY", "normal", "normal", latency_ms=100, llm_called=True, input_tokens=100, output_tokens=20),
        _record("b", "LLM_ONLY", "phishing", "phishing", latency_ms=200, llm_called=True, input_tokens=200, output_tokens=40),
        _record("a", "HYBRID", "normal", "normal", latency_ms=10, llm_called=False),
        _record("b", "HYBRID", "phishing", "phishing", latency_ms=220, llm_called=True, input_tokens=200, output_tokens=40),
    ]


def _metadata() -> dict:
    return {
        "source_split": "test",
        "dataset_fingerprint": "dataset",
        "evaluation_schema_version": 1,
        "model_id": "test-model",
        "region": "us-east-1",
        "prompt_version": "prompt-v1",
    }


def _policy() -> dict:
    return {
        "source_split": "validation",
        "selection": {
            "normal_probability_max": 0.1,
            "phishing_probability_min": 0.8,
            "llm_call_rate": 0.4,
            "target_recall": 0.95,
        },
    }


def _stacking_metadata() -> dict:
    return {
        "created_at": "2026-08-11T00:00:00+00:00",
        "model_sha256": "a" * 64,
        "schema_version": 1,
        "model": {"model_name": "stacking_phishing_classifier"},
    }


def _build(records: list[dict] | None = None) -> dict:
    return build_comparison_report(
        records or _records(),
        source_metadata=_metadata(),
        policy=_policy(),
        input_price_per_million=1.0,
        output_price_per_million=5.0,
        currency="USD",
        pricing_as_of="2026-08-13",
        stacking_metadata=_stacking_metadata(),
        generated_at="2026-08-13T00:00:00+00:00",
    )


def test_builds_three_mode_comparison_and_reductions() -> None:
    report = _build()
    modes = report["modes"]

    assert report["sample_count"] == 2
    assert report["report_schema_version"] == 3
    assert report["dataset"] == {
        "total_count": 2,
        "normal_count": 1,
        "phishing_count": 1,
        "positive_label": "phishing",
    }
    assert "every test sample" in report["classification_policy"][
        "full_dataset_metrics"
    ]
    assert (
        modes["SELF_MODEL_ONLY"]["classification"]["available_only"]["recall"]
        == 0.0
    )
    assert (
        modes["LLM_ONLY"]["classification"]["available_only"]["recall"]
        == 1.0
    )
    assert (
        modes["HYBRID"]["classification"]["available_only"]["f2"]
        == 1.0
    )
    assert (
        modes["HYBRID"]["classification"]["full_dataset"]["accuracy"]
        == 1.0
    )
    assert modes["SELF_MODEL_ONLY"]["operations"]["llm_call_rate"] == 0.0
    assert modes["LLM_ONLY"]["operations"]["llm_call_rate"] == 1.0
    assert modes["HYBRID"]["operations"]["llm_call_rate"] == 0.5
    assert modes["LLM_ONLY"]["cost"]["input_tokens"] == 300
    assert modes["HYBRID"]["cost"]["input_tokens"] == 200
    assert report["comparisons"]["llm_call_reduction_rate"] == pytest.approx(0.5)
    assert report["comparisons"]["hybrid_recall_delta_vs_self_model"] == 1.0
    assert report["routing_policy"]["test_uncertain_rate"] == 0.5


def test_records_unmeasured_llm_usage() -> None:
    records = _records()
    records[-1]["input_tokens"] = None
    report = _build(records)

    assert report["modes"]["HYBRID"]["cost"]["unmeasured_call_count"] == 1


def test_unknown_is_excluded_and_reported_as_unavailable() -> None:
    records = _records()
    records[-1].update(
        predicted_label="unknown",
        result_available=False,
        llm_available=False,
        all_engines_unavailable=True,
    )
    report = _build(records)
    hybrid = report["modes"]["HYBRID"]

    assert hybrid["availability"]["unavailable_count"] == 1
    assert hybrid["classification"]["available_only"]["sample_count"] == 1
    assert hybrid["classification"]["full_dataset"]["total_sample_count"] == 2
    assert hybrid["classification"]["full_dataset"]["unavailable_count"] == 1
    assert hybrid["classification"]["full_dataset"]["accuracy"] == 0.5
    assert (
        hybrid["classification"]["full_dataset"]["phishing_detection_rate"]
        == 0.0
    )


def test_full_dataset_metrics_use_all_records_in_each_mode() -> None:
    records = _records()
    for record in records:
        if record["mode"] == "LLM_ONLY" and record["sample_id"] == "a":
            record.update(
                predicted_label="unknown",
                result_available=False,
                llm_available=False,
                all_engines_unavailable=True,
            )

    report = _build(records)
    llm = report["modes"]["LLM_ONLY"]

    assert llm["classification"]["available_only"]["accuracy"] == 1.0
    assert llm["classification"]["full_dataset"]["accuracy"] == 0.5
    assert llm["classification"]["full_dataset"]["correct_count"] == 1
    assert llm["classification"]["full_dataset"]["unavailable_count"] == 1


def test_rejects_mismatched_sample_sets() -> None:
    records = _records()
    records.pop()

    with pytest.raises(ValueError, match="sample IDs differ"):
        _build(records)


def test_rejects_duplicate_mode_sample_record() -> None:
    records = _records()
    records.append(deepcopy(records[0]))

    with pytest.raises(ValueError, match="duplicate"):
        _build(records)


def test_rejects_test_derived_policy() -> None:
    policy = _policy()
    policy["source_split"] = "test"

    with pytest.raises(ValueError, match="validation"):
        build_comparison_report(
            _records(),
            source_metadata=_metadata(),
            policy=policy,
            input_price_per_million=1.0,
            output_price_per_million=5.0,
            currency="USD",
            pricing_as_of="2026-08-13",
            stacking_metadata=_stacking_metadata(),
        )


def test_rejects_invalid_price() -> None:
    with pytest.raises(ValueError):
        build_comparison_report(
            _records(),
            source_metadata=_metadata(),
            policy=_policy(),
            input_price_per_million=-1.0,
            output_price_per_million=5.0,
            currency="USD",
            pricing_as_of="2026-08-13",
            stacking_metadata=_stacking_metadata(),
        )


def test_renders_markdown_without_sample_identifiers() -> None:
    markdown = render_markdown_report(_build())

    assert "| Accuracy |" in markdown
    assert "Claude-only 대비 LLM 호출 감소율" in markdown
    assert "가격 기준일" in markdown
    assert "sample_id" not in markdown
    assert "| a |" not in markdown
    assert "Stacking artifact SHA-256" in markdown
    assert "목표 지표 충족 여부" in markdown
    assert "임계값 채택 결론" in markdown
    assert "Issue #37 PR 3 체크리스트" in markdown


def test_renders_reproducible_summary_csv_without_message_data() -> None:
    csv_report = render_csv_report(_build())

    assert "stacking_artifact_sha256" in csv_report
    assert "SELF_MODEL_ONLY" in csv_report
    assert "LLM_ONLY" in csv_report
    assert "HYBRID" in csv_report
    assert "sample_id" not in csv_report
    assert "message-0" not in csv_report
    assert "분석 대상" not in csv_report
