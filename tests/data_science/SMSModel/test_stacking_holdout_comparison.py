"""Stacking v1 vs v2 Holdout 비교 스크립트 테스트"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel import run_stacking_holdout_comparison as comparison


class DummyClassifier:
    """테스트용 더미 분류기"""

    def __init__(self, predictions: list[str], probabilities: list[float]) -> None:
        self._predictions = np.asarray(predictions)
        self._probabilities = np.asarray(probabilities)

    def predict(self, texts: Any) -> np.ndarray:
        return self._predictions[: len(texts)]

    def predict_proba(self, texts: Any) -> np.ndarray:
        prob_phishing = self._probabilities[: len(texts)]
        prob_normal = 1.0 - prob_phishing
        return np.column_stack([prob_normal, prob_phishing])


@pytest.mark.parametrize(
    ("v1_correct", "v2_correct", "expected"),
    [
        (True, True, "both_correct"),
        (False, False, "both_wrong"),
        (True, False, "v1_only_correct"),
        (False, True, "v2_only_correct"),
    ],
)
def test_classifies_pair_correctly(
    v1_correct: bool, v2_correct: bool, expected: str
) -> None:
    """쌍별 정답/오답 분류 로직 검증"""
    result = comparison.classify_pair(
        v1_correct=v1_correct, v2_correct=v2_correct
    )
    assert result == expected


def test_calculates_mcnemar_p_value_with_zero_discordant() -> None:
    """불일치 표본이 없을 때 p-value가 1.0을 반환하는지 검증"""
    assert comparison.calculate_mcnemar_p_value(0, 0) == 1.0


def test_calculates_mcnemar_p_value_significance() -> None:
    """비대칭/대칭 불일치 표본에 대한 McNemar p-value 계산 검증"""
    # 완전히 대칭인 경우 -> 유의하지 않음 (p=1.0)
    p_symmetric = comparison.calculate_mcnemar_p_value(10, 10)
    assert p_symmetric == pytest.approx(1.0)

    # 한쪽으로 치우친 경우 (0 vs 15) -> 유의함 (p < 0.05)
    p_asymmetric = comparison.calculate_mcnemar_p_value(0, 15)
    assert p_asymmetric < 0.05


def test_loads_model_from_joblib_payload(tmp_path: Path) -> None:
    """딕셔너리 페이로드 및 단일 객체 로드 검증"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # {"classifier": obj} 형태의 딕셔너리 아티팩트
    dict_model_path = model_dir / "dict_model.joblib"
    dummy_model = DummyClassifier(["normal"], [0.1])
    joblib.dump({"classifier": dummy_model, "schema_version": 1}, dict_model_path)

    loaded = comparison.load_model(dict_model_path)
    assert isinstance(loaded, DummyClassifier)

    # 파일이 없을 경우 FileNotFoundError 발생
    with pytest.raises(FileNotFoundError):
        comparison.load_model(model_dir / "non_existent.joblib")


def test_predict_model_extracts_phishing_probability() -> None:
    """예측 확률 및 레이블 추출 검증"""
    model = DummyClassifier(
        predictions=["normal", "phishing"], probabilities=[0.1, 0.85]
    )
    texts = pd.Series(["정상 텍스트", "피싱 텍스트"])

    probs, preds = comparison.predict_model(model, texts)

    assert len(probs) == 2
    assert len(preds) == 2
    assert probs[0] == pytest.approx(0.1)
    assert probs[1] == pytest.approx(0.85)
    assert list(preds) == ["normal", "phishing"]


def test_runs_comparison_end_to_end_and_generates_outputs(tmp_path: Path) -> None:
    """Holdout 비교 실행, CSV/JSON 산출물 생성 및 내용 검증"""
    # 테스트용 Holdout 데이터 준비
    holdout_path = tmp_path / "holdout_test.csv"
    holdout_df = pd.DataFrame(
        [
            # Sample 0: 정답 phishing -> v1 맞춤, v2 맞춤 (both_correct)
            {
                "text_fingerprint": "fp_0",
                "label": "phishing",
                "type": "loan",
                "text": "대출 안내",
            },
            # Sample 1: 정답 phishing -> v1 맞춤, v2 틀림 (v1_only_correct)
            {
                "text_fingerprint": "fp_1",
                "label": "phishing",
                "type": "delivery",
                "text": "택배 배송",
            },
            # Sample 2: 정답 normal   -> v1 틀림, v2 맞춤 (v2_only_correct)
            {
                "text_fingerprint": "fp_2",
                "label": "normal",
                "type": "receipt",
                "text": "결제 완료",
            },
            # Sample 3: 정답 normal   -> v1 틀림, v2 틀림 (both_wrong)
            {
                "text_fingerprint": "fp_3",
                "label": "normal",
                "type": "auth",
                "text": "인증 번호",
            },
        ]
    )
    holdout_df.to_csv(holdout_path, index=False, encoding="utf-8-sig")

    # v1, v2 모델 저장
    v1_model = DummyClassifier(
        predictions=["phishing", "phishing", "phishing", "phishing"],
        probabilities=[0.9, 0.8, 0.7, 0.6],
    )
    v2_model = DummyClassifier(
        predictions=["phishing", "normal", "normal", "phishing"],
        probabilities=[0.95, 0.2, 0.1, 0.8],
    )

    v1_path = tmp_path / "v1_model.joblib"
    v2_path = tmp_path / "v2_model.joblib"
    joblib.dump({"classifier": v1_model}, v1_path)
    joblib.dump({"classifier": v2_model}, v2_path)

    output_dir = tmp_path / "comparison_output"

    # 비교 실행
    summary = comparison.run_comparison(
        v1_model_path=v1_path,
        v2_model_path=v2_path,
        holdout_path=holdout_path,
        output_dir=output_dir,
    )

    # 요약 통계 검증
    assert summary["sample_count"] == 4
    assert summary["v1_accuracy"] == pytest.approx(0.5)  # 2/4
    assert summary["v2_accuracy"] == pytest.approx(0.5)  # 2/4

    ct = summary["contingency_table"]
    assert ct["both_correct"] == 1
    assert ct["both_wrong"] == 1
    assert ct["v1_only_correct"] == 1
    assert ct["v2_only_correct"] == 1

    assert summary["mcnemar_test"]["discordant_count"] == 2
    assert summary["mcnemar_test"]["p_value"] == pytest.approx(1.0)
    assert summary["mcnemar_test"]["is_significant_at_0_05"] is False

    # CSV 산출물 파일 검증
    csv_file = output_dir / "holdout_comparison_predictions.csv"
    assert csv_file.exists()

    result_df = pd.read_csv(csv_file)
    expected_columns = {
        "text_fingerprint",
        "label",
        "type",
        "v1_probability",
        "v1_prediction",
        "v1_correct",
        "v2_probability",
        "v2_prediction",
        "v2_correct",
        "comparison",
    }
    assert expected_columns.issubset(result_df.columns)
    assert list(result_df["comparison"]) == [
        "both_correct",
        "v1_only_correct",
        "v2_only_correct",
        "both_wrong",
    ]

    # JSON 산출물 파일 검증
    json_file = output_dir / "holdout_comparison_summary.json"
    assert json_file.exists()
    loaded_summary = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded_summary["sample_count"] == 4


def test_rejects_model_with_mismatched_checksum(tmp_path: Path) -> None:
    """metadata의 model_sha256과 파일 hash가 다르면 로드를 거부하는지 검증"""
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    model_path = model_dir / "model.joblib"
    joblib.dump(
        {"classifier": DummyClassifier(["normal"], [0.1])},
        model_path,
    )
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        comparison.load_model(model_path)


def test_accepts_model_with_matching_checksum(tmp_path: Path) -> None:
    """checksum이 일치하면 정상적으로 로드되는지 검증"""
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    model_path = model_dir / "model.joblib"
    joblib.dump(
        {"classifier": DummyClassifier(["normal"], [0.1])},
        model_path,
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {"model_sha256": comparison.calculate_sha256(model_path)}
        ),
        encoding="utf-8",
    )

    assert isinstance(comparison.load_model(model_path), DummyClassifier)


def test_requires_metadata_for_default_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """운영 artifact 기본 경로는 metadata 없이 로드되지 않는지 검증"""
    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {"classifier": DummyClassifier(["normal"], [0.1])},
        model_path,
    )
    monkeypatch.setattr(comparison, "DEFAULT_MODEL_PATHS", (model_path,))

    with pytest.raises(ValueError, match=r"metadata\.json is required"):
        comparison.load_model(model_path)

    # 테스트 fixture로 명시하면 permissive 경로가 유지됩니다.
    loaded = comparison.load_model(model_path, require_metadata=False)
    assert isinstance(loaded, DummyClassifier)


def test_predict_model_applies_artifact_threshold_in_fallback() -> None:
    """fallback 경로가 artifact threshold로 예측을 만드는지 검증"""
    model = DummyClassifier(
        predictions=["normal", "normal"],
        probabilities=[0.1, 0.85],
    )
    # 요약에 기록되는 threshold와 동일한 값이 예측에 적용되어야 합니다.
    model.threshold = 0.5

    _, predictions = comparison.predict_model(
        model, pd.Series(["정상 텍스트", "피싱 텍스트"])
    )

    assert list(predictions) == ["normal", "phishing"]


def test_load_holdout_excludes_training_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학습 pool과 겹치는 표본을 제외하고 감사 내역을 기록하는지 검증"""
    overlap_text = "겹치는 메시지"
    holdout_df = pd.DataFrame(
        [
            {"text": "정상 메시지", "label": "normal", "type": "chat"},
            {"text": "피싱 메시지", "label": "phishing", "type": "loan"},
            {"text": overlap_text, "label": "normal", "type": "chat"},
        ]
    )
    training_pool = pd.DataFrame(
        [
            {
                "text": overlap_text,
                "text_fingerprint": comparison.create_text_fingerprint(
                    comparison.normalize_text(overlap_text)
                ),
            }
        ]
    )

    monkeypatch.setattr(
        comparison, "EXPECTED_HOLDOUT_COUNT", len(holdout_df)
    )
    monkeypatch.setattr(
        comparison,
        "load_data",
        lambda path: (training_pool, holdout_df),
    )

    result = comparison.load_holdout()
    audit = result.attrs["holdout_audit"]

    assert len(result) == 2
    assert overlap_text not in set(result["text"])
    assert audit["source_row_count"] == 3
    assert audit["evaluated_row_count"] == 2
    assert audit["excluded_training_overlap_rows"] == 1
    assert audit["excluded_training_overlap_fingerprints"] == 1

