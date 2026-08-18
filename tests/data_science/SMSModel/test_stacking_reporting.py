"""Stacking v2 평가 보고서 생성 테스트"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.evaluation import (
    stacking_reporting as reporting,
)
from data_science.SMSModel.evaluation.metrics import (
    calculate_classification_metrics,
)


@pytest.fixture
def sample_evaluation_df() -> pd.DataFrame:
    """기본 테스트용 추론 결과 DataFrame fixture"""
    return pd.DataFrame(
        [
            # TP (피싱 -> 피싱)
            {
                "label": "phishing",
                "prediction": "phishing",
                "type": "smishing_loan",
                "text": "대출 안내 010-1234-5678",
                "text_fingerprint": "fp_tp_1",
                "latency": 10.0,
            },
            {
                "label": "phishing",
                "prediction": "phishing",
                "type": "smishing_delivery",
                "text": "택배 주소 확인 010-9999-8888",
                "text_fingerprint": "fp_tp_2",
                "latency": 20.0,
            },
            # FN (피싱 -> 정상: 미탐)
            {
                "label": "phishing",
                "prediction": "normal",
                "type": "smishing_loan",
                "text": "주민번호 900101-1234567 입력 요망",
                "text_fingerprint": "fp_fn_1",
                "latency": 15.0,
            },
            # TN (정상 -> 정상)
            {
                "label": "normal",
                "prediction": "normal",
                "type": "card_receipt",
                "text": "승인금액 50000원",
                "text_fingerprint": "fp_tn_1",
                "latency": 12.0,
            },
            {
                "label": "normal",
                "prediction": "normal",
                "type": "auth_code",
                "text": "인증번호 [123456]",
                "text_fingerprint": "fp_tn_2",
                "latency": 18.0,
            },
            # FP (정상 -> 피싱: 오탐)
            {
                "label": "normal",
                "prediction": "phishing",
                "type": "card_receipt",
                "text": "카드 재발급 계좌 110-123-456789",
                "text_fingerprint": "fp_fp_1",
                "latency": 25.0,
            },
        ]
    )


def test_generates_correct_overall_metrics_and_confusion_matrix(
    sample_evaluation_df: pd.DataFrame,
) -> None:
    """전체 지표(Accuracy, Precision, Recall, F1, F2) 및 혼동행렬 계산 검증."""
    report = reporting.generate_evaluation_report(sample_evaluation_df)
    metrics = report["overall_metrics"]
    cm = metrics["confusion_matrix"]

    # 혼동행렬 검증: TP=2, FN=1, TN=2, FP=1 (총 6건)
    assert cm["true_positive"] == 2
    assert cm["false_negative"] == 1
    assert cm["true_negative"] == 2
    assert cm["false_positive"] == 1

    # 지표 산출 검증
    # Accuracy = (2+2)/6 = 4/6 = 0.6667
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # Precision = 2/(2+1) = 2/3 = 0.6667
    assert metrics["precision"] == pytest.approx(2 / 3)
    # Recall = 2/(2+1) = 2/3 = 0.6667
    assert metrics["recall"] == pytest.approx(2 / 3)
    # F1 = 2/3
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    # F2 = (1 + 4) * (P * R) / (4*P + R) = 2/3
    assert metrics["f2_score"] == pytest.approx(2 / 3)


def test_calculates_phishing_and_normal_metrics_by_type(
    sample_evaluation_df: pd.DataFrame,
) -> None:
    """유형(type)별 세부 지표(Recall, FPR, 표본 수 등) 검증"""
    report = reporting.generate_evaluation_report(sample_evaluation_df)

    phishing_types = report["phishing_by_type"]
    normal_types = report["normal_by_type"]

    # 피싱 유형 검증 (smishing_loan: TP 1건, FN 1건 -> Recall 0.5)
    assert "smishing_loan" in phishing_types
    assert phishing_types["smishing_loan"]["sample_count"] == 2
    assert phishing_types["smishing_loan"]["true_positive"] == 1
    assert phishing_types["smishing_loan"]["false_negative"] == 1
    assert phishing_types["smishing_loan"]["recall"] == pytest.approx(0.5)

    # 피싱 유형 검증 (smishing_delivery: TP 1건 -> Recall 1.0)
    assert phishing_types["smishing_delivery"]["sample_count"] == 1
    assert phishing_types["smishing_delivery"]["recall"] == pytest.approx(1.0)

    # 정상 유형 검증 (card_receipt: TN 1건, FP 1건 -> FPR 0.5)
    assert "card_receipt" in normal_types
    assert normal_types["card_receipt"]["sample_count"] == 2
    assert normal_types["card_receipt"]["true_negative"] == 1
    assert normal_types["card_receipt"]["false_positive"] == 1
    assert normal_types["card_receipt"]["false_positive_rate"] == pytest.approx(0.5)

    # 정상 유형 검증 (auth_code: TN 1건 -> FPR 0.0)
    assert normal_types["auth_code"]["sample_count"] == 1
    assert normal_types["auth_code"]["false_positive_rate"] == pytest.approx(0.0)


def test_masks_sensitive_text_in_error_samples(
    sample_evaluation_df: pd.DataFrame,
) -> None:
    """오류(FP, FN) 샘플의 핑거프린트 유지 및 민감정보 마스킹 검증"""
    report = reporting.generate_evaluation_report(sample_evaluation_df)
    error_samples = report["error_samples"]

    fps = error_samples["false_positives"]
    fns = error_samples["false_negatives"]

    assert len(fps) == 1
    assert fps[0]["text_fingerprint"] == "fp_fp_1"
    # 원문 계좌번호가 마스킹되었는지 확인
    assert "110-123-456789" not in fps[0]["masked_text"]
    assert "[ACCOUNT/CARD]" in fps[0]["masked_text"]

    assert len(fns) == 1
    assert fns[0]["text_fingerprint"] == "fp_fn_1"
    # 원문 주민등록번호가 마스킹되었는지 확인
    assert "900101-1234567" not in fns[0]["masked_text"]
    assert "[RRN]" in fns[0]["masked_text"]


def test_calculates_latency_quantiles(
    sample_evaluation_df: pd.DataFrame,
) -> None:
    """Latency 통계 산출 검증"""
    report = reporting.generate_evaluation_report(sample_evaluation_df)
    latency_stats = report["latency_stats"]

    # latencies = [10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    expected_mean = float(np.mean([10.0, 12.0, 15.0, 18.0, 20.0, 25.0]))
    expected_p50 = float(np.percentile([10.0, 12.0, 15.0, 18.0, 20.0, 25.0], 50))
    expected_p95 = float(np.percentile([10.0, 12.0, 15.0, 18.0, 20.0, 25.0], 95))

    assert latency_stats["mean"] == pytest.approx(expected_mean)
    assert latency_stats["p50"] == pytest.approx(expected_p50)
    assert latency_stats["p95"] == pytest.approx(expected_p95)


def test_handles_missing_predictions_and_failures() -> None:
    """예외, 결측치, 엔진 실패 케이스 집계 및 지표 제외 검증"""
    df_with_failures = pd.DataFrame(
        [
            {"label": "phishing", "prediction": "phishing", "type": "loan", "has_exception": False},
            {"label": "phishing", "prediction": None, "type": "loan", "has_exception": False},
            {"label": "normal", "prediction": "error", "type": "auth", "has_exception": True},
            {"label": "normal", "prediction": "normal", "type": "auth", "has_exception": False},
        ]
    )

    report = reporting.generate_evaluation_report(df_with_failures)
    summary = report["sample_summary"]

    assert summary["total_samples"] == 4
    assert summary["successful_samples"] == 2
    assert summary["failure_counts"]["missing_result"] == 1
    assert summary["failure_counts"]["exception"] == 1
    assert summary["failure_counts"]["engine_failure"] == 2

    assert report["overall_metrics"]["accuracy"] == pytest.approx(1.0)


def test_masks_email_and_url_in_sensitive_text() -> None:
    """이메일과 URL이 마스킹되는지 검증"""
    masked = reporting.default_mask_sensitive_text(
        "문의는 alice@example.com 또는 https://www.baemin.me/qGa29G 로 주세요"
    )

    assert "alice@example.com" not in masked
    assert "[EMAIL]" in masked
    assert "baemin.me" not in masked
    assert "[URL]" in masked


def test_masks_datetime_before_account_pattern() -> None:
    """날짜/시각이 계좌번호로 오분류되지 않는지 검증"""
    masked = reporting.default_mask_sensitive_text("- 일시: 2026-11-29 19:56")

    assert "[DATETIME]" in masked
    assert "[ACCOUNT/CARD]" not in masked


def test_rejects_unsupported_labels() -> None:
    """지원하지 않는 label이 있으면 지표 계산 전에 거부하는지 검증"""
    df_with_unsupported_label = pd.DataFrame(
        [
            {"label": "spam", "prediction": "phishing", "type": "loan"},
            {"label": "normal", "prediction": "normal", "type": "auth"},
        ]
    )

    with pytest.raises(ValueError, match="unsupported labels"):
        reporting.generate_evaluation_report(df_with_unsupported_label)


def test_returns_empty_template_metrics_when_all_predictions_fail() -> None:
    """전건 실패 + template_group_id 조합에서 0으로 나누지 않는지 검증"""
    all_failed_df = pd.DataFrame(
        [
            {
                "label": "phishing",
                "prediction": None,
                "type": "loan",
                "template_group_id": "tg_1",
            },
            {
                "label": "normal",
                "prediction": "error",
                "type": "auth",
                "template_group_id": "tg_2",
            },
        ]
    )

    report = reporting.generate_evaluation_report(all_failed_df)
    template_metrics = report["unique_template_metrics"]

    assert report["sample_summary"]["successful_samples"] == 0
    assert report["overall_metrics"]["accuracy"] is None
    assert template_metrics["sample_count"] == 0
    assert template_metrics["accuracy"] is None
    assert template_metrics["precision"] is None
    assert template_metrics["recall"] is None
    assert template_metrics["f1_score"] is None
    assert template_metrics["f2_score"] is None


def test_saved_report_keeps_only_fingerprints(tmp_path: Path) -> None:
    """저장되는 보고서에 원문 파생 텍스트가 남지 않는지 검증"""
    test_df = pd.DataFrame(
        [
            {
                "label": "phishing",
                "type": "smishing_loan",
                "text": "대출 안내 010-1234-5678",
                "text_fingerprint": "fp_tp",
            },
            {
                "label": "normal",
                "type": "chat",
                "text": "오늘 저녁에 보자",
                "text_fingerprint": "fp_tn",
            },
            {
                "label": "normal",
                "type": "notice",
                "text": "문의는 alice@example.com 으로 주세요",
                "text_fingerprint": "fp_fp",
            },
        ]
    )
    predictions = np.asarray(["phishing", "normal", "phishing"])
    metrics = calculate_classification_metrics(
        test_df["label"].to_numpy(),
        predictions,
    )

    report = reporting.save_stacking_test_report(
        test_df=test_df,
        probabilities=np.asarray([0.9, 0.1, 0.8]),
        predictions=predictions,
        latencies_ms=[1.0, 2.0, 3.0],
        metrics=metrics,
        threshold=0.5,
        unavailable_models=(),
        output_directory=tmp_path,
        artifact_version="v3",
    )

    false_positives = report["error_samples"]["false_positives"]
    assert len(false_positives) == 1
    assert set(false_positives[0]) == {"text_fingerprint"}
    assert false_positives[0]["text_fingerprint"] == "fp_fp"

    raw_json = (tmp_path / "test_evaluation.json").read_text(encoding="utf-8")
    saved = json.loads(raw_json)
    for samples in saved["error_samples"].values():
        for sample in samples:
            assert set(sample) == {"text_fingerprint"}
    assert "masked_text" not in raw_json
    assert "alice@example.com" not in raw_json
    assert "저녁" not in raw_json
