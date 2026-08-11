"""Stacking 추론 서비스의 artifact 로딩 및 fail-safe 동작 테스트"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from app.analysis.text import stacking_analyzer as analyzer
from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)


@pytest.fixture(autouse=True)
def reset_stacking_model_cache():
    """테스트 사이에 전역 모델 캐시가 공유되지 않도록 초기화"""

    analyzer.reset_stacking_model_for_test()

    yield

    analyzer.reset_stacking_model_for_test()


def build_classifier() -> StackingPhishingClassifier:

    return StackingPhishingClassifier(
        n_splits=2,
        random_state=42,
        threshold=0.6,
    )


def test_loads_valid_artifact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    classifier = build_classifier()
    load_calls = 0

    def fake_load(_model_path):
        nonlocal load_calls
        load_calls += 1

        return {
            "schema_version": 1,
            "classifier": classifier,
        }

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        fake_load,
    )

    assert analyzer.is_stacking_model_loaded() is True
    assert analyzer.is_stacking_model_loaded() is True

    # 두 번째 호출에서는 전역 캐시를 사용해야 합니다.
    assert load_calls == 1


def test_loads_real_artifact_from_configured_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    classifier = build_classifier()
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(
        {"schema_version": 1, "classifier": classifier},
        model_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(
                    model_path.read_bytes()
                ).hexdigest()
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analyzer, "DEFAULT_STACKING_MODEL_PATH", model_path)

    assert analyzer.is_stacking_model_loaded() is True


def test_retries_after_transient_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = build_classifier()
    load_calls = 0

    def flaky_load(_model_path):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            raise OSError("temporary read failure")
        return {"schema_version": 1, "classifier": classifier}

    monkeypatch.setattr(analyzer.joblib, "load", flaky_load)

    assert analyzer.is_stacking_model_loaded() is False
    assert analyzer.is_stacking_model_loaded() is True
    assert load_calls == 2


def test_rejects_artifact_with_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"tampered")
    model_path.with_name("metadata.json").write_text(
        json.dumps({"model_sha256": "0" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setattr(analyzer, "DEFAULT_STACKING_MODEL_PATH", model_path)

    assert analyzer.is_stacking_model_loaded() is False


def test_missing_artifact_returns_fail_safe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def raise_file_not_found(_model_path):
        raise FileNotFoundError("missing stacking artifact")

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        raise_file_not_found,
    )

    result = analyzer.analyze_text_with_stacking(
        "즉시 계좌로 송금하세요."
    )

    assert result == {
        "engine": "stacking",
        "is_available": False,
        "result": {
            "risk_score": None,
            "risk_probability": None,
            "confidence": 0.0,
            "is_suspected_phishing": None,
            "threshold": None,
            "model_scores": {},
            "unavailable_models": [],
            "error_message": "Stacking Model Unavailable",
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        # payload 자체가 dictionary가 아닌 경우
        None,
        [],
        "invalid artifact",
        # 지원하지 않는 schema version
        {
            "schema_version": 999,
            "classifier": None,
        },
        # classifier 타입이 잘못된 경우
        {
            "schema_version": 1,
            "classifier": object(),
        },
    ],
)
def test_invalid_artifact_returns_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    payload,
) -> None:

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        lambda _model_path: payload,
    )

    assert analyzer.is_stacking_model_loaded() is False

    result = analyzer.analyze_text_with_stacking("테스트 메시지")

    assert result["engine"] == "stacking"
    assert result["is_available"] is False
    assert result["result"]["risk_score"] is None
    assert result["result"]["is_suspected_phishing"] is None
    assert (
        result["result"]["error_message"]
        == "Stacking Model Unavailable"
    )


def test_returns_successful_stacking_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    classifier = build_classifier()

    monkeypatch.setattr(
        classifier,
        "predict_one",
        lambda _text: SimpleNamespace(
            risk_probability=0.87,
            risk_score=87,
            confidence=0.74,
            is_suspected_phishing=True,
            threshold=0.6,
            model_scores={
                "naive_bayes": 0.81,
                "logistic_regression": 0.84,
                "linear_svm": 0.78,
            },
            unavailable_models=(),
        ),
    )

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        lambda _model_path: {
            "schema_version": 1,
            "classifier": classifier,
        },
    )

    result = analyzer.analyze_text_with_stacking(
        "즉시 계좌로 송금하세요."
    )

    assert result == {
        "engine": "stacking",
        "is_available": True,
        "result": {
            "risk_score": 87,
            "risk_probability": 0.87,
            "confidence": 0.74,
            "is_suspected_phishing": True,
            "threshold": 0.6,
            "model_scores": {
                "naive_bayes": 0.81,
                "logistic_regression": 0.84,
                "linear_svm": 0.78,
            },
            "unavailable_models": [],
            "error_message": None,
        },
    }


def test_reports_unavailable_base_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    classifier = build_classifier()

    monkeypatch.setattr(
        classifier,
        "predict_one",
        lambda _text: SimpleNamespace(
            risk_probability=0.91,
            risk_score=91,
            confidence=0.82,
            is_suspected_phishing=True,
            threshold=0.6,
            model_scores={
                "naive_bayes": 0.9,
                # 실패 모델은 중립값으로 표현됩니다.
                "logistic_regression": 0.5,
                "linear_svm": 0.88,
            },
            unavailable_models=("logistic_regression",),
        ),
    )

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        lambda _model_path: {
            "schema_version": 1,
            "classifier": classifier,
        },
    )

    result = analyzer.analyze_text_with_stacking(
        "계좌가 정지됩니다. 즉시 확인하세요."
    )

    assert result["is_available"] is True
    assert result["result"]["risk_score"] == 91
    assert result["result"]["is_suspected_phishing"] is True
    assert result["result"]["unavailable_models"] == [
        "logistic_regression"
    ]


def test_inference_error_returns_fail_safe_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    classifier = build_classifier()

    def raise_inference_error(_text):
        raise RuntimeError("expected inference failure")

    monkeypatch.setattr(
        classifier,
        "predict_one",
        raise_inference_error,
    )

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        lambda _model_path: {
            "schema_version": 1,
            "classifier": classifier,
        },
    )

    result = analyzer.analyze_text_with_stacking(
        "즉시 송금하세요."
    )

    assert result == {
        "engine": "stacking",
        "is_available": False,
        "result": {
            "risk_score": None,
            "risk_probability": None,
            "confidence": 0.0,
            "is_suspected_phishing": None,
            "threshold": None,
            "model_scores": {},
            "unavailable_models": [],
            "error_message": "Stacking Inference Error",
        },
    }


def test_rejects_non_string_input() -> None:

    with pytest.raises(TypeError, match="text must be a string"):
        analyzer.analyze_text_with_stacking(123)  # type: ignore[arg-type]


def test_raw_message_is_not_written_to_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:

    classifier = build_classifier()
    raw_message = "주민번호 900101-1234567 즉시 전송하세요"

    monkeypatch.setattr(
        classifier,
        "predict_one",
        lambda _text: SimpleNamespace(
            risk_probability=0.95,
            risk_score=95,
            confidence=0.9,
            is_suspected_phishing=True,
            threshold=0.6,
            model_scores={
                "naive_bayes": 0.94,
                "logistic_regression": 0.93,
                "linear_svm": 0.96,
            },
            unavailable_models=(),
        ),
    )

    monkeypatch.setattr(
        analyzer.joblib,
        "load",
        lambda _model_path: {
            "schema_version": 1,
            "classifier": classifier,
        },
    )

    with caplog.at_level(
        logging.INFO,
        logger=analyzer.__name__,
    ):
        analyzer.analyze_text_with_stacking(raw_message)

    assert raw_message not in caplog.text
    assert "900101-1234567" not in caplog.text
