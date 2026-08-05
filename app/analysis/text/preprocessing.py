"""SMS 모델의 학습과 추론에서 공통으로 사용하는 전처리 로직."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# 학습된 모델은 아래 피처 순서에 의존하므로 순서를 변경하면 안 됩니다.
STRUCT_FEATURE_NAMES: tuple[str, ...] = (
    "has_url",
    "has_short_url",
    "has_phone",
    "has_amount",
    "has_web_tag",
    "is_long_text",
)

URL_PATTERN = re.compile(
    r"(?i)(?<!@)(?:https?://|www\.)"
    r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+;=%]+"
)
RRN_PATTERN = re.compile(r"(?<!\d)\d{6}[- ]\d{7}(?!\d)")
CARD_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}"
    r"|\d{4}[- ]?\d{6}[- ]?\d{5})(?!\d)"
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}|0\d{9,10})(?!\d)"
)
ACCOUNT_PATTERN = re.compile(
    r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,6}(?:-\d{1,6})?(?!\d)"
    r"|(?<!\d)\d{10,14}(?!\d)"
)
EMAIL_PATTERN = re.compile(r"(?i)[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}")
AMOUNT_PATTERN = re.compile(r"\d+[,\d]*원")
FORMAT_ARTIFACT_PATTERN = re.compile(r"={2,}|■|□|▪|▫|●|○|\s-\s|\s:\s")
SHORT_URL_PATTERN = re.compile(
    r"(?i)bit\.ly|goo\.gl|tinyurl|gourl|ow\.ly|n\.bnuee|han\.gl|cutt\.ly"
)
WEB_TAG_PATTERN = re.compile(r"\[Web발신\]|\[국외발신\]|\[국제발신\]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def mask_pii(text: str) -> str:
    """개인정보를 Spring PiiMaskingService와 같은 순서로 마스킹합니다."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    masked = RRN_PATTERN.sub("[RRN]", text)
    masked = CARD_PATTERN.sub("[CARD]", masked)
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    masked = ACCOUNT_PATTERN.sub("[ACCOUNT]", masked)
    masked = EMAIL_PATTERN.sub("[EMAIL]", masked)
    return masked


def normalize_text(text: str) -> str:
    """URL·개인정보·금액을 치환하고 공백을 정리한 모델 입력을 만듭니다."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    parts: list[str] = []
    last_end = 0

    # URL 내부 숫자가 전화번호나 계좌번호로 오인되지 않도록 URL부터 분리합니다.
    for match in URL_PATTERN.finditer(text):
        parts.append(mask_pii(text[last_end:match.start()]))
        parts.append("[URL]")
        last_end = match.end()

    parts.append(mask_pii(text[last_end:]))
    normalized = "".join(parts)
    normalized = AMOUNT_PATTERN.sub("[AMOUNT]", normalized)
    normalized = FORMAT_ARTIFACT_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def extract_struct_features(text: str, *, has_url: bool | None = None) -> list[int]:
    """단일 원문 SMS에서 고정 순서의 구조 피처 6개를 추출합니다."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    detected_url = bool(URL_PATTERN.search(text))
    url_feature = detected_url if has_url is None else bool(has_url)

    return [
        int(url_feature),
        int(bool(SHORT_URL_PATTERN.search(text))),
        int(bool(PHONE_PATTERN.search(text) or "[PHONE]" in text)),
        int(bool(AMOUNT_PATTERN.search(text) or "[AMOUNT]" in text)),
        int(bool(WEB_TAG_PATTERN.search(text))),
        int(len(text) > 100),
    ]


def extract_struct_feature_matrix(
    texts: Iterable[str] | pd.Series,
    has_urls: Iterable[bool] | pd.Series | None = None,
) -> np.ndarray:
    """여러 원문 SMS의 구조 피처를 ``(n_samples, 6)`` 배열로 반환합니다."""
    text_list = list(texts)

    if has_urls is None:
        url_list: list[bool | None] = [None] * len(text_list)
    else:
        url_list = list(has_urls)
        if len(text_list) != len(url_list):
            raise ValueError(
                "texts and has_urls must contain the same number of items"
            )

    rows = [
        extract_struct_features(text, has_url=has_url)
        for text, has_url in zip(text_list, url_list, strict=True)
    ]

    if not rows:
        return np.empty((0, len(STRUCT_FEATURE_NAMES)), dtype=np.int8)
    return np.asarray(rows, dtype=np.int8)
