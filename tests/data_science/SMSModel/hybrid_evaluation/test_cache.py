"""Claude test cache 보안 및 재현성 테스트."""
from __future__ import annotations

import json

import pytest

from data_science.SMSModel.hybrid_evaluation.cache import (
    ClaudeTestCache,
    calculate_dataset_fingerprint,
)

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def build_cache(tmp_path) -> ClaudeTestCache:
    return ClaudeTestCache(
        path=tmp_path / "llm_test_predictions.json",
        dataset_fingerprint="dataset-sha256",
        model_id="test-claude-haiku",
        region="us-east-1",
        prompt_version="smishing-v1:prompt-sha256",
    )


def successful_analysis() -> dict:
    return {
        "provider": "AWS_BEDROCK",
        "model_id": "test-claude-haiku",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
        },
        "latency_ms": 123.4,
        "result": {
            "grade": "DANGEROUS",
            "risk_score": 85,
            "tone_analysis": "민감할 수 있는 응답",
            "evidence": ["캐시에 저장하면 안 됨"],
            "reason": "캐시에 저장하면 안 됨",
            "error_message": None,
        },
    }


def test_dataset_fingerprint_is_order_independent() -> None:
    first = calculate_dataset_fingerprint(
        [
            (FINGERPRINT_A, "normal"),
            (FINGERPRINT_B, "phishing"),
        ]
    )

    second = calculate_dataset_fingerprint(
        [
            (FINGERPRINT_B, "phishing"),
            (FINGERPRINT_A, "normal"),
        ]
    )

    assert first == second


def test_dataset_fingerprint_changes_when_label_changes() -> None:
    first = calculate_dataset_fingerprint(
        [(FINGERPRINT_A, "normal")]
    )

    second = calculate_dataset_fingerprint(
        [(FINGERPRINT_A, "phishing")]
    )

    assert first != second


def test_cache_stores_only_safe_fields(tmp_path) -> None:
    cache = build_cache(tmp_path)

    cache.store_analysis(
        text_fingerprint=FINGERPRINT_A,
        analysis=successful_analysis(),
    )

    cache.save()

    serialized = (
        tmp_path / "llm_test_predictions.json"
    ).read_text(encoding="utf-8")

    assert "민감할 수 있는 응답" not in serialized
    assert "캐시에 저장하면 안 됨" not in serialized
    assert "tone_analysis" not in serialized
    assert "evidence" not in serialized
    assert "reason" not in serialized
    assert "text" not in json.loads(serialized)["predictions"][0]

    entry = cache.entries[FINGERPRINT_A]

    assert entry["grade"] == "DANGEROUS"
    assert entry["risk_score"] == 85
    assert entry["input_tokens"] == 100
    assert entry["output_tokens"] == 20


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("dataset_fingerprint", "different-dataset"),
        ("model_id", "different-model"),
        ("prompt_version", "different-prompt"),
        ("evaluation_schema_version", 999),
        ("source_split", "validation"),
    ],
)
def test_rejects_incompatible_cache(
    tmp_path,
    field_name,
    changed_value,
) -> None:
    cache = build_cache(tmp_path)

    cache.store_analysis(
        text_fingerprint=FINGERPRINT_A,
        analysis=successful_analysis(),
    )

    cache.save()

    path = tmp_path / "llm_test_predictions.json"
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload[field_name] = changed_value

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="metadata mismatch",
    ):
        build_cache(tmp_path).load()


def test_require_complete_rejects_missing_predictions(
    tmp_path,
) -> None:
    cache = build_cache(tmp_path)

    cache.store_analysis(
        text_fingerprint=FINGERPRINT_A,
        analysis=successful_analysis(),
    )

    with pytest.raises(
        RuntimeError,
        match="incomplete",
    ):
        cache.require_complete(
            {
                FINGERPRINT_A,
                FINGERPRINT_B,
            }
        )


@pytest.mark.asyncio
async def test_cached_analysis_matches_llm_contract(
    tmp_path,
) -> None:
    cache = build_cache(tmp_path)

    cache.store_analysis(
        text_fingerprint=FINGERPRINT_A,
        analysis=successful_analysis(),
    )

    analysis = await cache.analyze_cached(
        FINGERPRINT_A
    )

    assert analysis["provider"] == "AWS_BEDROCK"
    assert analysis["is_cached"] is True
    assert analysis["model_id"] == "test-claude-haiku"
    assert analysis["result"]["grade"] == "DANGEROUS"
    assert analysis["result"]["risk_score"] == 85
    assert analysis["usage"]["input_tokens"] == 100


def test_entries_property_does_not_expose_mutable_cache_state(tmp_path) -> None:
    cache = build_cache(tmp_path)
    cache.store_analysis(
        text_fingerprint=FINGERPRINT_A,
        analysis=successful_analysis(),
    )

    exported = cache.entries
    exported[FINGERPRINT_A]["risk_score"] = 0

    assert cache.entries[FINGERPRINT_A]["risk_score"] == 85


def test_rejects_non_hash_fingerprint(tmp_path) -> None:
    cache = build_cache(tmp_path)

    with pytest.raises(ValueError, match="SHA-256"):
        cache.store_analysis(
            text_fingerprint="SMS 원문을 식별자로 사용하면 안 됨",
            analysis=successful_analysis(),
        )
