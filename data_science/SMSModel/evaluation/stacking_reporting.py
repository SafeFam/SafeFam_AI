"""Stacking v2 종합 평가 보고서 생성 모듈"""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from data_science.SMSModel.evaluation.metrics import ClassificationMetrics
from data_science.SMSModel.modeling.stacking import StackingPhishingClassifier


EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RRN_PATTERN = re.compile(r"\b\d{6}[-.\s]?[1-4]\d{6}\b")
# 날짜/시각은 계좌번호 규칙보다 먼저 치환해야 "2026-11-29 19:56"이
# [ACCOUNT/CARD]로 잘못 분류되지 않습니다.
DATETIME_PATTERN = re.compile(
    r"\b\d{4}\s?[-./]\s?\d{1,2}\s?[-./]\s?\d{1,2}\.?"
    r"(?:\s?(?:오전|오후)?\s?\d{1,2}:\d{2}(?::\d{2})?)?"
)
PHONE_PATTERN = re.compile(r"\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b")
ACCOUNT_PATTERN = re.compile(r"\b(?:\d[-.\s]?){10,16}\b")


def default_mask_sensitive_text(text: str) -> str:
    """이메일, URL, 전화번호, 주민번호, 계좌번호 등 민감정보 마스킹"""
    if not isinstance(text, str):
        return ""

    # 이메일
    text = EMAIL_PATTERN.sub("[EMAIL]", text)

    # URL
    text = URL_PATTERN.sub("[URL]", text)

    # 주민등록번호
    text = RRN_PATTERN.sub("[RRN]", text)

    # 날짜/시각 (계좌번호 규칙보다 먼저 적용)
    text = DATETIME_PATTERN.sub("[DATETIME]", text)

    # 전화번호
    text = PHONE_PATTERN.sub("[PHONE]", text)

    # 계좌번호/카드번호
    text = ACCOUNT_PATTERN.sub("[ACCOUNT/CARD]", text)

    return text


def calculate_f_beta(
    precision: Optional[float], recall: Optional[float], beta: float = 1.0
) -> Optional[float]:
    """F-beta score 계산 (F1: beta=1.0, F2: beta=2.0)"""
    if beta <= 0:
        raise ValueError("beta must be greater than zero")
    if precision is None or recall is None:
        return None
    beta_sq = beta**2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * (precision * recall) / denominator


def calculate_wilson_interval(
    successes: int,
    sample_count: int,
    *,
    z_score: float = 1.96,
) -> dict[str, float] | None:
    """이항 비율의 Wilson 95% 신뢰구간을 반환합니다."""
    if sample_count == 0:
        return None
    if successes < 0 or successes > sample_count:
        raise ValueError("successes must be between zero and sample_count")
    proportion = successes / sample_count
    denominator = 1 + z_score**2 / sample_count
    center = (
        proportion + z_score**2 / (2 * sample_count)
    ) / denominator
    margin = (
        z_score
        * np.sqrt(
            proportion * (1 - proportion) / sample_count
            + z_score**2 / (4 * sample_count**2)
        )
        / denominator
    )
    return {
        "lower": max(0.0, float(center - margin)),
        "upper": min(1.0, float(center + margin)),
    }


def generate_evaluation_report(
    result_df: pd.DataFrame,
    mask_func: Optional[Callable[[str], str]] = None,
    max_error_samples: int = 50,
) -> Dict[str, Any]:
    """모델 평가 보고서 생성"""
    required_columns = {"label", "prediction", "type"}
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise ValueError(
            f"result_df is missing columns: {sorted(missing_columns)}"
        )
    if result_df.empty:
        raise ValueError("result_df must not be empty")
    if max_error_samples < 0:
        raise ValueError("max_error_samples must not be negative")

    if mask_func is None:
        mask_func = default_mask_sensitive_text

    total_samples = len(result_df)

    # 실패 및 예외 건수 집계
    missing_pred_mask = result_df["prediction"].isna()
    exception_mask = result_df.get(
        "has_exception", pd.Series(False, index=result_df.index)
    )
    failed_mask = (
        missing_pred_mask
        | exception_mask
        | result_df["prediction"].isin(["error", "fail", None])
    )

    missing_result_count = int(missing_pred_mask.sum())
    exception_count = int(exception_mask.sum())
    engine_failure_count = int(failed_mask.sum())

    # 유효한 평가 대상 데이터셋
    valid_df = result_df[~failed_mask].copy()
    successful_samples = len(valid_df)

    unsupported_predictions = (
        set(valid_df["prediction"].astype(str))
        - {"normal", "phishing"}
    )
    if unsupported_predictions:
        raise ValueError(
            "result_df contains unsupported predictions: "
            f"{sorted(unsupported_predictions)}"
        )

    unsupported_labels = (
        set(valid_df["label"].astype(str))
        - {"normal", "phishing"}
    )
    if unsupported_labels:
        raise ValueError(
            "result_df contains unsupported labels: "
            f"{sorted(unsupported_labels)}"
        )

    # 전체 혼동 행렬 (Confusion Matrix)
    tp = int(
        (
            (valid_df["label"] == "phishing")
            & (valid_df["prediction"] == "phishing")
        ).sum()
    )
    fn = int(
        (
            (valid_df["label"] == "phishing")
            & (valid_df["prediction"] == "normal")
        ).sum()
    )
    tn = int(
        (
            (valid_df["label"] == "normal")
            & (valid_df["prediction"] == "normal")
        ).sum()
    )
    fp = int(
        (
            (valid_df["label"] == "normal")
            & (valid_df["prediction"] == "phishing")
        ).sum()
    )

    # 전체 평가 지표 (Accuracy, Precision, Recall, F1, F2)
    accuracy = (tp + tn) / successful_samples if successful_samples > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = calculate_f_beta(precision, recall, beta=1.0)
    f2 = calculate_f_beta(precision, recall, beta=2.0)

    # 유형별 피싱 집계
    phishing_rows = valid_df[valid_df["label"] == "phishing"]
    phishing_by_type = {}

    for type_name, group in phishing_rows.groupby("type", dropna=False):
        type_str = "unknown" if pd.isna(type_name) else str(type_name)
        t_tp = int((group["prediction"] == "phishing").sum())
        t_fn = int((group["prediction"] == "normal").sum())
        sample_count = len(group)

        phishing_by_type[type_str] = {
            "sample_count": sample_count,
            "true_positive": t_tp,
            "false_negative": t_fn,
            "recall": t_tp / sample_count if sample_count > 0 else None,
            "recall_95_ci": calculate_wilson_interval(
                t_tp,
                sample_count,
            ),
        }

    # 유형별 정상 집계
    normal_rows = valid_df[valid_df["label"] == "normal"]
    normal_by_type = {}

    for type_name, group in normal_rows.groupby("type", dropna=False):
        type_str = "unknown" if pd.isna(type_name) else str(type_name)
        t_tn = int((group["prediction"] == "normal").sum())
        t_fp = int((group["prediction"] == "phishing").sum())
        sample_count = len(group)

        normal_by_type[type_str] = {
            "sample_count": sample_count,
            "true_negative": t_tn,
            "false_positive": t_fp,
            "false_positive_rate": (
                t_fp / sample_count if sample_count > 0 else None
            ),
            "false_positive_rate_95_ci": calculate_wilson_interval(
                t_fp,
                sample_count,
            ),
        }

    # FP / FN 에러 샘플 추출
    def extract_error_samples(df_subset: pd.DataFrame) -> List[Dict[str, str]]:
        samples = []
        for _, row in df_subset.head(max_error_samples).iterrows():
            samples.append(
                {
                    "text_fingerprint": row.get("text_fingerprint", ""),
                    "masked_text": mask_func(row.get("text", ""))[:200],
                }
            )
        return samples

    fp_rows = valid_df[
        (valid_df["label"] == "normal")
        & (valid_df["prediction"] == "phishing")
    ]
    fn_rows = valid_df[
        (valid_df["label"] == "phishing") & (valid_df["prediction"] == "normal")
    ]

    error_samples = {
        "false_positives": extract_error_samples(fp_rows),
        "false_negatives": extract_error_samples(fn_rows),
    }

    # Latency 지표 (평균, P50, P95)
    latency_stats = {
        "sample_count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
    }
    latency_column = (
        "latency_ms" if "latency_ms" in result_df.columns else "latency"
    )
    if latency_column in result_df.columns:
        latencies = pd.to_numeric(
            result_df[latency_column], errors="coerce"
        ).dropna()
        if not latencies.empty:
            latency_stats = {
                "sample_count": len(latencies),
                "mean": float(latencies.mean()),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
            }

    unique_template_metrics = None
    if "template_group_id" in valid_df.columns:
        unique_rows = valid_df.drop_duplicates(
            subset="template_group_id",
            keep="first",
        )
        unique_sample_count = len(unique_rows)
        unique_tp = int(
            (
                (unique_rows["label"] == "phishing")
                & (unique_rows["prediction"] == "phishing")
            ).sum()
        )
        unique_fn = int(
            (
                (unique_rows["label"] == "phishing")
                & (unique_rows["prediction"] == "normal")
            ).sum()
        )
        unique_tn = int(
            (
                (unique_rows["label"] == "normal")
                & (unique_rows["prediction"] == "normal")
            ).sum()
        )
        unique_fp = int(
            (
                (unique_rows["label"] == "normal")
                & (unique_rows["prediction"] == "phishing")
            ).sum()
        )
        unique_precision = (
            unique_tp / (unique_tp + unique_fp)
            if unique_tp + unique_fp
            else None
        )
        unique_recall = (
            unique_tp / (unique_tp + unique_fn)
            if unique_tp + unique_fn
            else None
        )
        unique_template_metrics = {
            "sample_count": unique_sample_count,
            # 모든 예측이 실패하면 유효 행이 없으므로 overall_metrics와 동일하게
            # accuracy를 None으로 둡니다.
            "accuracy": (
                (unique_tp + unique_tn) / unique_sample_count
                if unique_sample_count > 0
                else None
            ),
            "precision": unique_precision,
            "recall": unique_recall,
            "f1_score": calculate_f_beta(
                unique_precision,
                unique_recall,
            ),
            "f2_score": calculate_f_beta(
                unique_precision,
                unique_recall,
                beta=2.0,
            ),
            "true_positive": unique_tp,
            "false_negative": unique_fn,
            "true_negative": unique_tn,
            "false_positive": unique_fp,
        }

    # 최종 보고서 객체 조합
    return {
        "overall_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f2_score": f2,
            "confusion_matrix": {
                "true_positive": tp,
                "false_negative": fn,
                "true_negative": tn,
                "false_positive": fp,
            },
        },
        "sample_summary": {
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "failure_counts": {
                "missing_result": missing_result_count,
                "exception": exception_count,
                "engine_failure": engine_failure_count,
            },
        },
        "phishing_by_type": phishing_by_type,
        "normal_by_type": normal_by_type,
        "unique_template_metrics": unique_template_metrics,
        "error_samples": error_samples,
        "latency_stats": latency_stats,
    }


def predict_probabilities_with_latency(
    classifier: StackingPhishingClassifier,
    evaluation_df: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...], list[float]]:
    """한 번의 표본 순회로 확률, 가용성 및 단건 지연시간을 수집합니다."""
    probabilities: list[float] = []
    unavailable_models: set[str] = set()
    durations_ms: list[float] = []
    for row_index in range(len(evaluation_df)):
        row = evaluation_df.iloc[row_index : row_index + 1]
        started_at = perf_counter_ns()
        row_probabilities, row_unavailable = (
            classifier.predict_probabilities(row)
        )
        durations_ms.append(
            (perf_counter_ns() - started_at) / 1_000_000
        )
        if len(row_probabilities) != 1:
            raise ValueError("single-row inference must return one probability")
        probabilities.append(float(row_probabilities[0]))
        unavailable_models.update(row_unavailable)

    return (
        np.asarray(probabilities, dtype=float),
        tuple(sorted(unavailable_models)),
        durations_ms,
    )


def save_stacking_test_report(
    *,
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    latencies_ms: list[float],
    metrics: ClassificationMetrics,
    threshold: float,
    unavailable_models: tuple[str, ...],
    output_directory: Path,
) -> Dict[str, Any]:
    """개인정보를 제거한 test 평가 JSON과 Markdown을 저장합니다."""
    if not (
        len(test_df)
        == len(probabilities)
        == len(predictions)
        == len(latencies_ms)
    ):
        raise ValueError(
            "test rows, probabilities, and predictions must align"
        )

    result_df = test_df.copy()
    result_df["probability"] = np.asarray(probabilities, dtype=float)
    result_df["prediction"] = np.asarray(predictions, dtype=str)
    result_df["latency_ms"] = latencies_ms

    report = generate_evaluation_report(result_df)

    # 커밋되는 보고서에는 원문에서 파생된 문자열을 남기지 않고
    # 재현 가능한 fingerprint만 유지합니다.
    report["error_samples"] = {
        category: [
            {"text_fingerprint": sample.get("text_fingerprint", "")}
            for sample in samples
        ]
        for category, samples in report["error_samples"].items()
    }

    report.update(
        {
            "schema_version": 1,
            "split": "test",
            "threshold": float(threshold),
            "unavailable_models": list(unavailable_models),
            "shared_metrics": metrics.to_dict(),
        }
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "test_evaluation.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    overall = report["overall_metrics"]
    confusion = overall["confusion_matrix"]
    latency = report["latency_stats"]
    markdown = "\n".join(
        [
            "# Stacking v2 Test Evaluation",
            "",
            "Threshold selection: validation only; final metrics: test only.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Samples | {report['sample_summary']['total_samples']} |",
            f"| Accuracy | {overall['accuracy']:.6f} |",
            f"| Precision | {overall['precision']:.6f} |",
            f"| Recall | {overall['recall']:.6f} |",
            f"| F1 | {overall['f1_score']:.6f} |",
            f"| F2 | {overall['f2_score']:.6f} |",
            f"| TN | {confusion['true_negative']} |",
            f"| FP | {confusion['false_positive']} |",
            f"| FN | {confusion['false_negative']} |",
            f"| TP | {confusion['true_positive']} |",
            f"| Mean latency (ms) | {latency['mean']:.3f} |",
            f"| P50 latency (ms) | {latency['p50']:.3f} |",
            f"| P95 latency (ms) | {latency['p95']:.3f} |",
            "",
        ]
    )
    (output_directory / "test_evaluation.md").write_text(
        markdown,
        encoding="utf-8",
    )
    return report
