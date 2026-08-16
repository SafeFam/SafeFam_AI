"""하이브리드 평가 실행 결과의 provenance 저장 테스트."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_science.SMSModel import run_hybrid_evaluation as evaluation
from data_science.SMSModel.hybrid_evaluation import (
    EvaluationMode,
    EvaluationRecord,
)


def _record() -> EvaluationRecord:
    return EvaluationRecord(
        sample_id="sha256:test",
        mode=EvaluationMode.SELF_MODEL_ONLY,
        expected_label="phishing",
        predicted_label="phishing",
        latency_ms=1.0,
        result_available=True,
        llm_called=False,
        llm_available=False,
        fallback_applied=False,
        all_engines_unavailable=False,
        decision_source="STACKING",
        routing_decision=None,
        routing_reason=None,
        error_code=None,
        llm_provider=None,
        llm_model=None,
        input_tokens=None,
        output_tokens=None,
    )


def test_saves_reproducible_evaluation_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sms_split_test.csv"
    manifest_content = b"text_fingerprint,split\nabc,test\n"
    manifest_path.write_bytes(manifest_content)
    output_path = tmp_path / "evaluation_records.json"

    monkeypatch.setattr(
        evaluation,
        "SPLIT_MANIFEST_PATH",
        manifest_path,
    )
    monkeypatch.setattr(
        evaluation,
        "EVALUATION_RECORDS_PATH",
        output_path,
    )

    evaluation.save_evaluation_records(
        [_record()],
        dataset_fingerprint="dataset-fingerprint",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["split_manifest"] == manifest_path.name
    assert payload["split_manifest_sha256"] == hashlib.sha256(
        manifest_content
    ).hexdigest()
    assert payload["random_state"] == 42
    assert payload["positive_label"] == "phishing"
    assert payload["source_split"] == "test"
    assert payload["record_count"] == 1


def test_requires_split_manifest_before_saving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "SPLIT_MANIFEST_PATH",
        tmp_path / "missing.csv",
    )

    try:
        evaluation.save_evaluation_records(
            [_record()],
            dataset_fingerprint="dataset-fingerprint",
        )
    except FileNotFoundError as exception:
        assert "split manifest is required" in str(exception)
    else:
        raise AssertionError("missing split manifest must be rejected")
