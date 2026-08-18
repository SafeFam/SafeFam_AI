"""Stacking v2 종합 평가 보고서 생성 모듈"""
import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


def default_mask_sensitive_text(text: str) -> str:
    """전화번호, 주민번호, 계좌번호 등 민감정보 마스킹"""
    if not isinstance(text, str):
        return ""
    
    # 주민등록번호 
    text = re.sub(r"\b\d{6}[-.\s]?[1-4]\d{6}\b", "[RRN]", text)
    
    # 전화번호
    text = re.sub(r"\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "[PHONE]", text)
    
    # 계좌번호/카드번호 
    text = re.sub(r"\b(?:\d[-.\s]?){10,16}\b", "[ACCOUNT/CARD]", text)
    
    return text


def calculate_f_beta(
    precision: Optional[float], recall: Optional[float], beta: float = 1.0
) -> Optional[float]:
    """F-beta score 계산 (F1: beta=1.0, F2: beta=2.0)"""
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    beta_sq = beta**2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


def generate_evaluation_report(
    result_df: pd.DataFrame,
    mask_func: Optional[Callable[[str], str]] = None,
    max_error_samples: int = 50,
) -> Dict[str, Any]:
    """모델 평가 보고서 생성"""
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
    latency_stats = {"mean": None, "p50": None, "p95": None}
    if "latency" in result_df.columns:
        latencies = result_df["latency"].dropna()
        if not latencies.empty:
            latency_stats = {
                "mean": float(latencies.mean()),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
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
        "error_samples": error_samples,
        "latency_stats": latency_stats,
    }