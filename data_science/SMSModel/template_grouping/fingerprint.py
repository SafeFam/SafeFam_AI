"""정규화 SMS의 fingerprint 생성과 완전 중복 처리"""

from __future__ import annotations

import hashlib

import pandas as pd


def create_text_fingerprint(text_norm: str) -> str:
    """정규화 텍스트에서 결정적인 SHA-256 fingerprint를 생성"""
    if not isinstance(text_norm, str):
        raise TypeError("text_norm must be a string")

    canonical_text = text_norm.strip()
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def add_text_fingerprints(
    df: pd.DataFrame,
    *,
    text_column: str = "text_norm",
) -> pd.DataFrame:
    """원본을 변경하지 않고 text_fingerprint 컬럼 추가"""
    if text_column not in df.columns:
        raise ValueError(f"missing text column: {text_column}")

    result = df.copy()
    result["text_fingerprint"] = result[text_column].apply(create_text_fingerprint)
    return result


def validate_duplicate_labels(
    df: pd.DataFrame,
    *,
    fingerprint_column: str = "text_fingerprint",
    label_column: str = "label",
) -> None:
    """동일한 정규화 메시지에 서로 다른 label이 있으면 중단"""
    required_columns = {fingerprint_column, label_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    label_counts = df.groupby(fingerprint_column)[label_column].nunique()
    conflicting = label_counts[label_counts > 1].index
    if conflicting.empty:
        return

    example_columns = [fingerprint_column, label_column]
    if "text_norm" in df.columns:
        example_columns.append("text_norm")
    examples = (
        df[df[fingerprint_column].isin(conflicting)][example_columns]
        .head(10)
        .to_dict(orient="records")
    )
    raise ValueError(
        f"identical normalized messages contain conflicting labels: {examples}"
    )


def remove_exact_duplicates(
    df: pd.DataFrame,
    *,
    fingerprint_column: str = "text_fingerprint",
) -> pd.DataFrame:
    """동일 fingerprint 중 첫 번째 행만 남김"""
    if fingerprint_column not in df.columns:
        raise ValueError(f"missing fingerprint column: {fingerprint_column}")

    return df.drop_duplicates(subset=fingerprint_column, keep="first").reset_index(
        drop=True
    )
