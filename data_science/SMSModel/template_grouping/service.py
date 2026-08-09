"""SMS template grouping 처리 순서를 조율하는 공개 서비스"""

from __future__ import annotations

import pandas as pd

from .config import TemplateGroupingConfig
from .fingerprint import (
    add_text_fingerprints,
    remove_exact_duplicates,
    validate_duplicate_labels,
)
from .similarity import assign_template_groups


def prepare_template_groups(
    df: pd.DataFrame,
    *,
    config: TemplateGroupingConfig | None = None,
    text_column: str = "text_norm",
    label_column: str = "label",
) -> pd.DataFrame:
    """fingerprint, 충돌 검사, 중복 제거, 유사 그룹화를 순서대로 수행"""
    result = add_text_fingerprints(df, text_column=text_column)
    validate_duplicate_labels(
        result,
        fingerprint_column="text_fingerprint",
        label_column=label_column,
    )
    result = remove_exact_duplicates(
        result,
        fingerprint_column="text_fingerprint",
    )
    return assign_template_groups(
        result,
        config=config,
        text_column=text_column,
    )
