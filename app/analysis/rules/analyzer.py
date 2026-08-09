import logging
import re

logger = logging.getLogger(__name__)

# 각 신호별 배점 (합산 후 100점 만점으로 캡)
MALICIOUS_DOMAIN_RULE_SCORE = 100  # 로컬 가드 도메인 룰(.ru, testsafebrowsing) 매치 -> 단독으로 만점(기존 동작 유지)
ACCOUNT_NUMBER_SCORE = 30
CARD_NUMBER_SCORE = 30
INSTITUTION_MENTION_SCORE = 15
URGENCY_KEYWORD_SCORE_PER_CATEGORY = 10
URGENCY_KEYWORD_SCORE_CAP = 30

# 로컬 가드 도메인 룰 (기존 scan_service.py에 있던 것을 이관)
_RE_MALICIOUS_DOMAIN_HINT = re.compile(r"\.ru\b|testsafebrowsing", re.IGNORECASE)

# 계좌번호: 하이픈으로 구분된 은행 계좌 형식(예: 110-1234-567890) 또는 10~14자리 연속 숫자.
# 한글 조사(으로/에게 등)가 숫자 바로 뒤에 공백 없이 붙는 경우가 많아, 일반 \b 대신
# "숫자가 아닌 것"만 확인하는 lookaround를 사용 (\b는 한글도 단어문자로 취급해 조사 앞에서 걸리지 않음)
_RE_ACCOUNT_NUMBER = re.compile(
    r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,8}(?!\d)|(?<!\d)\d{10,14}(?!\d)"
)

# 카드번호: 4자리씩 4묶음(하이픈/공백 구분 또는 연속 16자리)
_RE_CARD_NUMBER = re.compile(
    r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)|(?<!\d)\d{16}(?!\d)"
)

# 금융감독원 등록 기준 주요 금융기관/공공기관 명칭 (사칭 대상으로 흔히 악용되는 목록)
FINANCIAL_INSTITUTIONS = [
    # 은행
    "국민은행",
    "KB국민은행",
    "신한은행",
    "우리은행",
    "하나은행",
    "KEB하나은행",
    "NH농협은행",
    "농협은행",
    "IBK기업은행",
    "기업은행",
    "SC제일은행",
    "한국씨티은행",
    "씨티은행",
    "케이뱅크",
    "카카오뱅크",
    "토스뱅크",
    "수협은행",
    "산업은행",
    "KDB산업은행",
    "새마을금고",
    "신협",
    "우체국",
    # 카드사
    "신한카드",
    "삼성카드",
    "현대카드",
    "롯데카드",
    "하나카드",
    "KB국민카드",
    "국민카드",
    "우리카드",
    "BC카드",
    "비씨카드",
    "NH농협카드",
    "씨티카드",
    # 증권/보험
    "미래에셋증권",
    "삼성증권",
    "한국투자증권",
    "NH투자증권",
    "키움증권",
    "삼성생명",
    "교보생명",
    "한화생명",
    # 공공/사법기관 (사칭 빈도가 높은 기관)
    "금융감독원",
    "금융위원회",
    "검찰청",
    "경찰청",
    "국세청",
    "관세청",
    "법원",
    "대법원",
    "건강보험공단",
    "국민건강보험공단",
    "우정사업본부",
]

# 카테고리별 금융 긴급/강압 키워드 (카테고리당 최초 1회만 카운트)
URGENCY_KEYWORD_CATEGORIES = {
    "이체/송금": ["이체", "송금", "입금 확인", "출금"],
    "대출": ["대출 승인", "대출 실행", "대환대출", "저금리 대출"],
    "계좌 상태": ["계좌 정지", "계좌 동결", "지급 정지", "출금 정지"],
    "명의/개인정보": ["명의도용", "명의 도용", "개인정보 유출", "개인정보 노출"],
    "카드 상태": ["카드 정지", "카드 도용", "부정 사용"],
    "압류/연체": ["압류", "연체", "체납"],
    "긴급 확인 요구": ["즉시 확인", "즉시 조치", "지금 바로 확인"],
}


def _check_malicious_domain(traced_url: str | None) -> bool:
    if not traced_url:
        return False
    return bool(_RE_MALICIOUS_DOMAIN_HINT.search(traced_url))


def _check_institution_mention(text: str) -> bool:
    return any(name in text for name in FINANCIAL_INSTITUTIONS)


def _check_urgency_keywords(text: str) -> list[str]:
    matched_categories = []
    for category, keywords in URGENCY_KEYWORD_CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            matched_categories.append(category)
    return matched_categories


# 금감원 금융기관 명칭 DB 대조 + 금융 키워드 가중치 + 계좌/카드번호 패턴 탐지를 결합한 로컬 규칙 기반 트랙.
# 나이브 베이즈/Gemini와 달리 결정론적 규칙만으로 판정하며, URL이 있으면 로컬 가드 도메인 룰도 함께 검사한다.
def analyze_text_with_rules(text: str, traced_url: str | None = None) -> dict:
    matched_rules: list[str] = []
    score = 0

    has_malicious_domain = _check_malicious_domain(traced_url)
    if has_malicious_domain:
        matched_rules.append(f"로컬 가드 도메인 룰 매치 ({traced_url})")
        score += MALICIOUS_DOMAIN_RULE_SCORE

    if _RE_ACCOUNT_NUMBER.search(text) or "[ACCOUNT]" in text:
        matched_rules.append("계좌번호로 추정되는 숫자 패턴 발견")
        score += ACCOUNT_NUMBER_SCORE

    if _RE_CARD_NUMBER.search(text) or "[CARD]" in text:
        matched_rules.append("카드번호로 추정되는 숫자 패턴 발견")
        score += CARD_NUMBER_SCORE

    if _check_institution_mention(text):
        matched_rules.append("금융기관/공공기관 명칭 언급")
        score += INSTITUTION_MENTION_SCORE

    urgency_categories = _check_urgency_keywords(text)
    if urgency_categories:
        urgency_score = min(
            len(urgency_categories) * URGENCY_KEYWORD_SCORE_PER_CATEGORY,
            URGENCY_KEYWORD_SCORE_CAP,
        )
        matched_rules.append(f"금융 긴급 키워드 매치: {', '.join(urgency_categories)}")
        score += urgency_score

    rule_score = min(score, 100)

    if matched_rules:
        logger.info(
            f"[RuleEngine] 규칙 매치 - 점수: {rule_score} | 사유: {matched_rules}"
        )

    return {
        "rule_score": rule_score,
        "matched_rules": matched_rules,
        # URL 트랙 자체의 악성 판정에도 영향을 주는 로컬 가드 도메인 룰 매치 여부 (scan_service에서 사용)
        "has_malicious_domain_pattern": has_malicious_domain,
    }
