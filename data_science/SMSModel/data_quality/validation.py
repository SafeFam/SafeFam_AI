"""SMS 원본 데이터의 schema와 taxonomy 무결성 검증"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .taxonomy import (
    ALLOWED_MESSAGE_TYPES,
    is_allowed_label_type_pair,
    normalize_legacy_message_type,
)


REQUIRED_DATASET_COLUMNS = {
    "text",
    "label",
    "type",
    "has_url",
    "source",
}

def normalize_message_types(
        dataset: pd.DataFrame,
) -> pd.DataFrame:
    """기존 type alias를 canonical type으로 정규화"""

    if "type" not in dataset.columns:
        raise ValueError("dataset is missing type column")

    result = dataset.copy()
    result["type"] = result["type"].map(
        normalize_legacy_message_type
    )

    return result

def validate_dataset_schema(
        dataset: pd.DataFrame,
) -> None:
    """필수 컬럼, 결측치 및 기본 자료형 검증"""

    missing = REQUIRED_DATASET_COLUMNS - set(
        dataset.columns
    )

    if missing:
        raise ValueError(
            f"dataset is missing required columns: {missing}"
        )

    if dataset.empty:
        raise ValueError("dataset must not be empty")

    if dataset[list(REQUIRED_DATASET_COLUMNS)].isna().any().any():
        raise ValueError(
            "dataset contains missing required values"
        )

    if not dataset["text"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    ).all():
        raise ValueError(
            "text must contain non-empty strings"
        )

def validate_message_types(
        dataset: pd.DataFrame,
) -> None:
    """허용 type과 laebl/type 조합을 검증"""

    observed_types = set(
        dataset["type"].astype(str)
    )
    unsupported = observed_types - ALLOWED_MESSAGE_TYPES

    if unsupported:
        raise ValueError(
            f"dataset contains unsupported types: "
            f"{sorted(unsupported)}"
        )

    invalid_rows = dataset[
        ~dataset.apply(
            lambda row: is_allowed_label_type_pair(
                str(row["label"]),
                str(row["type"]),
            ),
            axis=1,
        )
    ]

    if not invalid_rows.empty:
        examples = invalid_rows[
            ["label", "type", "source"]
        ].head(10).to_dict(orient="records")

        raise ValueError(
            "dataset contains invalid label/type pairs: "
            f"{examples}"
        )

def validate_sources(
        dataset: pd.DataFrame,
        *,
        allowed_sources: Iterable[str],
) -> None:
    """알 수 없는 provenance 값을 차단"""

    allowed = set(allowed_sources)
    observed = set(dataset["source"].astype(str))
    unsupported = observed - allowed

    if unsupported:
        raise ValueError(
            f"dataset contains unsupported sources: "
            f"{sorted(unsupported)}"
        )

def validate_sms_dataset(
    dataset: pd.DataFrame,
    *,
    allowed_sources: Iterable[str],    
) -> pd.DataFrame:
    """정규화된 복사본을 반환하며 전체 무결성 검증"""

    validate_dataset_schema(dataset)

    normalized = normalize_message_types(dataset)

    validate_message_types(normalized)
    validate_sources(
        normalized,
        allowed_sources=allowed_sources,
    )

    return normalized