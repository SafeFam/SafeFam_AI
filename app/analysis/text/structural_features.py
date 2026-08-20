"""Stacking 모델에서 사용하는 구조적 피싱 특징 추출기"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.analysis.text.preprocessing import (
    AMOUNT_PATTERN,
    CARD_PATTERN,
    PHONE_PATTERN,
    SHORT_URL_PATTERN,
    URL_PATTERN,
    WEB_TAG_PATTERN,
    contains_account_number,
)


# 자격증명을 가리키는 표현
CREDENTIAL_TERMS = (
    r"개인정보|주민(?:등록)?번호|신분증|인증번호|"
    r"비밀번호|보안카드|OTP|공동인증서|계좌번호"
)

# 제3자에게 넘기라는 요구
HANDOVER_TERMS = r"알려|불러|전송|회신|보내|공유|말씀|제공|드리면"

# 개인정보 및 인증정보 "요구" 표현
PERSONAL_INFO_PATTERN = re.compile(
    rf"(?:{CREDENTIAL_TERMS})[^.!?\n]{{0,20}}?(?:{HANDOVER_TERMS})",
    re.IGNORECASE,
)

# 송금 및 결제 요구 표현
TRANSFER_REQUEST_PATTERN = re.compile(
    r"송금|입금|이체|결제|납부|대납|"
    r"돈\s*(?:보내|부쳐)|금액을?\s*(?:보내|입금)",
    re.IGNORECASE,
)

# 긴급한 행동을 유도하는 표현
URGENCY_PATTERN = re.compile(
    r"긴급|즉시|지금\s*바로|오늘\s*안에|"
    r"금일\s*내|마감|기한|정지\s*예정|"
    r"압류|체납|연체|명의도용",
    re.IGNORECASE,
)

# 링크 클릭을 직접 요구하는 표현
LINK_ACTION_PATTERN = re.compile(
    r"링크|URL|주소|접속|클릭|확인하기|"
    r"아래\s*(?:주소|링크)|사이트에서",
    re.IGNORECASE,
)


# 합법 광고 문자의 법정 표기
AD_DISCLOSURE_PATTERN = re.compile(
    r"\(\s*광고\s*\)|\[\s*광고\s*\]|^광고",
    re.IGNORECASE | re.MULTILINE,
)

OPT_OUT_PATTERN = re.compile(
    r"수신\s*거부|무료거부|"
    r"080[-.\s]?\d{3,4}[-.\s]?\d{4}",
    re.IGNORECASE,
)


# 배열의 열 순서가 학습 및 추론에서 동일해야 하므로 상수로 고정
STACKING_STRUCTURAL_FEATURE_NAMES: tuple[str, ...] = (
    "has_url",
    "has_short_url",
    "has_phone",
    "has_account",
    "has_card",
    "has_amount",
    "has_web_tag",
    "has_urgency",
    "has_transfer_request",
    "has_personal_info_request",
    "has_link_action",
    "is_long_text",
    "has_ad_disclosure",
    "has_opt_out",
)


@dataclass(frozen=True)
class StructuralFeatureResult:
    """단일 메시지에서 추출한 구조 특징 결과"""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError("structural feature values must be one-dimensional")

        if len(values) != len(self.names):
            raise ValueError("feature names and values must have the same length")

        if not np.isfinite(values).all():
            raise ValueError("structural feature values must be finite")

        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError("structural feature values must be normalized to 0..1")

        object.__setattr__(self, "values", values)


def extract_stacking_structural_features(
    text: str,
) -> StructuralFeatureResult:
    """단일 SMS에서 0 또는 1로 정규화된 구조 특징을 추출"""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    values = np.asarray(
        [
            bool(URL_PATTERN.search(text)),
            bool(SHORT_URL_PATTERN.search(text)),
            bool(PHONE_PATTERN.search(text) or "[PHONE]" in text),
            bool(contains_account_number(text) or "[ACCOUNT]" in text),
            bool(CARD_PATTERN.search(text) or "[CARD]" in text),
            bool(AMOUNT_PATTERN.search(text) or "[AMOUNT]" in text),
            bool(WEB_TAG_PATTERN.search(text)),
            bool(URGENCY_PATTERN.search(text)),
            bool(TRANSFER_REQUEST_PATTERN.search(text)),
            bool(PERSONAL_INFO_PATTERN.search(text)),
            bool(LINK_ACTION_PATTERN.search(text)),
            len(text) > 100,
            bool(AD_DISCLOSURE_PATTERN.search(text)),
            bool(OPT_OUT_PATTERN.search(text)),
        ],
        dtype=np.float64,
    )

    return StructuralFeatureResult(
        values=values,
        names=STACKING_STRUCTURAL_FEATURE_NAMES,
    )


def extract_stacking_structural_matrix(
    texts: Iterable[str],
) -> np.ndarray:
    """복수 메시지를 meta-classifier 입력 행렬로 변환"""

    text_list = list(texts)

    if not text_list:
        return np.empty(
            (0, len(STACKING_STRUCTURAL_FEATURE_NAMES)),
            dtype=np.float64,
        )

    rows = [
        extract_stacking_structural_features(text).values
        for text in text_list
    ]

    return np.vstack(rows)