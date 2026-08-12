"""LLM validation 캐시 입력 및 메타데이터 검증 테스트."""

import json

import pytest

from data_science.SMSModel import run_hybrid_threshold_selection as selection


@pytest.mark.parametrize("score", [True, False])
def test_rejects_boolean_llm_scores(score: bool) -> None:
    assert selection._is_available_llm_result(
        score=score,
        grade="DANGEROUS",
        error_message=None,
    ) is False


def test_accepts_integer_llm_score() -> None:
    assert selection._is_available_llm_result(
        score=85,
        grade="DANGEROUS",
        error_message=None,
    ) is True


def _cache_payload() -> dict:
    return {
        "schema_version": 2,
        "provider": "AWS_BEDROCK",
        "model_id": selection.settings.BEDROCK_MODEL_ID,
        "region": selection.settings.AWS_REGION,
        "prompt_version": "smishing-v1",
        "predictions": [],
    }


@pytest.mark.parametrize(
    ("field", "runtime_field", "message"),
    [
        ("model_id", "BEDROCK_MODEL_ID", "LLM cache model mismatch"),
        ("region", "AWS_REGION", "LLM cache region mismatch"),
    ],
)
def test_rejects_cache_runtime_mismatch(
    tmp_path,
    monkeypatch,
    field: str,
    runtime_field: str,
    message: str,
) -> None:
    cache_path = tmp_path / "llm_validation_predictions.json"
    payload = _cache_payload()
    payload[field] = f"{getattr(selection.settings, runtime_field)}-mismatch"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(selection, "LLM_VALIDATION_CACHE_PATH", cache_path)

    with pytest.raises(ValueError, match=message):
        selection._load_cached_predictions()
