"""세 모드 비교 보고서 집계 및 렌더링 테스트."""
from __future__ import annotations

from copy import deepcopy

import pytest

from data_science.SMSModel.hybrid_evaluation import (
    build_comparison_report,
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
        },
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
        generated_at="2026-08-13T00:00:00+00:00",
    )


def test_builds_three_mode_comparison_and_reductions() -> None:
    report = _build()
    modes = report["modes"]

    assert report["sample_count"] == 2
    assert modes["SELF_MODEL_ONLY"]["classification"]["recall"] == 0.0
    assert modes["LLM_ONLY"]["classification"]["recall"] == 1.0
    assert modes["HYBRID"]["classification"]["f2"] == 1.0
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
    assert hybrid["classification"]["sample_count"] == 1


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
        )


def test_renders_markdown_without_sample_identifiers() -> None:
    markdown = render_markdown_report(_build())

    assert "| Accuracy |" in markdown
    assert "Claude-only 대비 LLM 호출 감소율" in markdown
    assert "가격 기준일" in markdown
    assert "sample_id" not in markdown
    assert "| a |" not in markdown
