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
HANDOVER_TERMS = (
    r"(?:알려|불러|전송|회신|보내|공유|말씀|제공)\s*"
    r"(?:주(?:세요|시면|시기|십시오|시길|셔야|시오)|해\s*주(?:세요|시면|십시오)"
    r"|바랍니다|바람|요망|부탁|하세요|하십시오|줘|달라)"
)

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

# 긴급한 행동을 유도하는 표현.
# "마감"/"기한"은 원래 포함됐으나, 정상 이용안내문(사용기한·신청마감 등)에도
# 흔히 등장해 오탐을 키웠다 - 정상 유형(택배·카드결제·공공기관 등) 실측에서
# has_urgency가 FP 그룹에서 4배 이상 더 흔했다. 진짜 압박 표현만 남긴다.
URGENCY_PATTERN = re.compile(
    r"긴급|즉시|지금\s*바로|오늘\s*안에|"
    r"금일\s*내|정지\s*예정|"
    r"압류|체납|연체|명의도용",
    re.IGNORECASE,
)

# 링크 클릭을 직접 요구하는 표현
LINK_ACTION_PATTERN = re.compile(
    r"링크|URL|주소|접속|클릭|확인하기|"
    r"아래\s*(?:주소|링크)|사이트에서",
    re.IGNORECASE,
)


# 사칭에 동원되는 친족 호칭
KINSHIP_TERMS = (
    r"엄마|아빠|어머니|아버지|딸|아들|형수|장모|장인|처형|"
    r"누나|오빠|언니|삼촌|이모|고모|사위|며느리"
)

# 연락 수단이 바뀐 사정을 설명하는 표현
CONTACT_TROUBLE_TERMS = (
    r"임시\s*번호|이\s*번호로|폰|핸드폰|휴대폰|액정|수리|대리점|"
    r"고장|깨(?:져|졌)|분실|인증\s*이?\s*안|공인인증서"
)

# 본인 대신 송금해 달라는 요구
PROXY_TRANSFER_TERMS = (
    r"(?:대신|먼저|저대신)[^\n]{0,20}?(?:보내|이체|송금|입금)"
)

# 가족을 사칭해 접근하는 표현
FAMILY_IMPERSONATION_PATTERN = re.compile(
    rf"(?:{KINSHIP_TERMS})[^\n]{{0,100}}?(?:{CONTACT_TROUBLE_TERMS})"
    rf"|(?:{CONTACT_TROUBLE_TERMS})[^\n]{{0,100}}?(?:{KINSHIP_TERMS})"
    rf"|(?:{KINSHIP_TERMS})[^\n]{{0,100}}?{PROXY_TRANSFER_TERMS}"
)

# 통신사 문자를 벗어나 통제 밖 대화방으로 유도하는 표현
CHATROOM_INVITE_PATTERN = re.compile(
    r"(?:카\s*톡|카카오\s*톡|오픈\s*채팅|채팅\s*방|정보\s*방|밴드|텔레그램)"
    r"[^\n]{0,25}?(?:아이디|ID|입장|참여|초대|추가|코드|오세요|놀러)"
    r"|(?:입장|참여)\s*(?:코드|하시면)|저희\s*방",
    re.IGNORECASE,
)

PLUS_FRIEND_PATTERN = re.compile(r"플러스\s*친구")

# 손쉬운 고수익을 내세운 채용 및 부업 유인
JOB_OFFER_LURE_PATTERN = re.compile(
    r"(?:재택|알바|아르바이트|부업|투잡|모집|채용)"
    r"[^\n]{0,120}?"
    r"(?:시급|일급|일당|고수익|추가\s*수입|급여|예치금|"
    r"(?:하루|일)\s*[\d,\-~]+\s*만)"
)

# 종목 추천이나 수익률을 내세운 투자 유인
INVESTMENT_LURE_PATTERN = re.compile(
    r"(?:종목|급등|폭등|수익률|상한가|우량주|주도주|코인|이더|리딩)"
    r"[^\n]{0,80}?(?:무료|공개|추천|당첨|입장|참여|방|수익|드리)"
    r"|(?:무료|단독)[^\n]{0,30}?(?:종목|급등|공개)"
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
    "has_family_impersonation",
    "has_chatroom_invite",
    "has_job_offer_lure",
    "has_investment_lure",
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
            bool(FAMILY_IMPERSONATION_PATTERN.search(text)),
            bool(
                CHATROOM_INVITE_PATTERN.search(text)
                and not PLUS_FRIEND_PATTERN.search(text)
            ),
            bool(JOB_OFFER_LURE_PATTERN.search(text)),
            bool(INVESTMENT_LURE_PATTERN.search(text)),
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