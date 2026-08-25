import re
from enum import Enum


class PhishingType(str, Enum):
    """SafeFam_BE의 PhishingCategory와 공유하는 메시징 계약."""

    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    LOAN = "LOAN"
    JOB = "JOB"
    DELIVERY = "DELIVERY"
    MESSENGER = "MESSENGER"
    OTHER = "OTHER"


# SafeFam_BE PhishingCategoryClassifier와 순서 및 키워드를 동일하게 유지한다.
# 여러 유형의 일치 횟수가 같으면 먼저 등록된 유형을 선택한다.
_CATEGORY_PATTERNS: tuple[tuple[PhishingType, re.Pattern[str]], ...] = (
    (
        PhishingType.FINANCIAL_INSTITUTION,
        re.compile(
            "은행|금융감독원|금감원|카드사|카드|계좌|금융기관",
            re.IGNORECASE,
        ),
    ),
    (
        PhishingType.GOVERNMENT_AGENCY,
        re.compile(
            "검찰|경찰|국세청|건강보험|공단|법원|수사관|검사|공무원",
            re.IGNORECASE,
        ),
    ),
    (
        PhishingType.LOAN,
        re.compile("대출|저금리|대환|한도|신용등급|보증료|상환", re.IGNORECASE),
    ),
    (
        PhishingType.JOB,
        re.compile("채용|구인|알바|아르바이트|재택근무|고수익|업무", re.IGNORECASE),
    ),
    (
        PhishingType.DELIVERY,
        re.compile("택배|배송|운송장|주소지|우체국|반송", re.IGNORECASE),
    ),
    (
        PhishingType.MESSENGER,
        re.compile(
            "카카오톡|카톡|메신저|휴대폰 고장|엄마|아빠|자녀|친구 추가",
            re.IGNORECASE,
        ),
    ),
)


def classify_phishing_type(
    content: str,
    *,
    sender: str | None = None,
) -> PhishingType:
    """발신자와 문자에서 가장 많이 언급된 피싱 주제를 반환한다."""

    analysis_text = f"{(sender or '').strip()} {content.strip()}"
    selected = PhishingType.OTHER
    highest_match_count = 0

    for phishing_type, pattern in _CATEGORY_PATTERNS:
        match_count = sum(1 for _ in pattern.finditer(analysis_text))
        if match_count > highest_match_count:
            highest_match_count = match_count
            selected = phishing_type

    return selected
