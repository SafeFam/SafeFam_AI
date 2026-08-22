import logging
import re
from urllib.parse import urlparse

from app.analysis.institution.registry import OFFICIAL_INSTITUTIONS, OfficialInstitution
from app.analysis.text.preprocessing import PHONE_PATTERN

logger = logging.getLogger(__name__)

# GSB/VT 블랙리스트는 신고 이력이 쌓여야 걸리므로 아직 신고되지 않은 신규 피싱 도메인은
# 놓칠 수 있다. 이 검사는 "그 기관의 공식 도메인이 아니다"라는 사실만으로 신고 이력과
# 무관하게 즉시 잡아낼 수 있어 단독으로도 강한 신호로 취급한다.
INSTITUTION_MISMATCH_SCORE = 50

# URL이 없는 문자(발신번호 사칭형 등)에도 적용 가능한 신호라 도메인 대조와 별도로 채점한다.
INSTITUTION_PHONE_MISMATCH_SCORE = 40

# PHONE_PATTERN은 "0"으로 시작하는 일반/휴대전화만 잡는다. 대표번호(1588/1599/1661 등
# 15xx~19xx 국번 없는 착신과금 번호)는 앞자리가 0이 아니라서 별도 패턴이 필요하다.
# 하이픈/공백 구분자를 요구해 "182명" 같은 우연한 숫자열을 전화번호로 오인하지 않게 한다.
_RE_TOLL_FREE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{2}[- ]\d{4}(?!\d)")

# SMS 발신 표기 관례상 발신 주체는 "[국민은행]"처럼 대괄호 태그로 표기되는 경우가 대부분이다.
# 본문 전체에서 기관명을 찾으면 "보내신 분: 농협하나로마트" 같은, 발신 주체가 아니라 단순히
# 상호명 등으로 스쳐 지나가는 언급까지 사칭으로 오인해 정상 문자(택배 발신자명 등)를 오탐하므로,
# 실제 발신 주체를 나타내는 대괄호 태그 안으로 범위를 좁힌다.
_RE_BRACKET_TAG = re.compile(r"\[([^\[\]]+)\]")


def _extract_domain(url: str) -> str | None:
    # 잘못된 형식의 URL(예: 괄호가 안 닫힌 IPv6 authority)은 urlparse가 ValueError를 던질 수 있다.
    # 이 함수는 규칙 엔진 미리보기 단계(traced_url 확정 전)에서도 호출되므로, 여기서 직접
    # 방어해 호출부의 예외 처리에 기대지 않고 안전하게 "판정 불가(None)"로 처리한다.
    try:
        hostname = urlparse(url).hostname
    except ValueError:
        return None
    return hostname.lower() if hostname else None


def _is_official_domain(domain: str, official_domains: tuple[str, ...]) -> bool:
    return any(
        domain == official or domain.endswith(f".{official}") for official in official_domains
    )


def _find_mentioned_institution(text: str) -> OfficialInstitution | None:
    tags = _RE_BRACKET_TAG.findall(text)
    for institution in OFFICIAL_INSTITUTIONS:
        if any(alias in tag for tag in tags for alias in institution.aliases):
            return institution
    return None


def _normalize_phone(number: str) -> str:
    return re.sub(r"\D", "", number)


def _extract_phone_numbers(text: str) -> list[str]:
    candidates = PHONE_PATTERN.findall(text) + _RE_TOLL_FREE_PHONE.findall(text)
    # 순서를 보존하면서 중복(정규화 기준)을 제거
    seen: set[str] = set()
    numbers: list[str] = []
    for candidate in candidates:
        normalized = _normalize_phone(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            numbers.append(candidate)
    return numbers


def _matches_official_phone(number: str, official_phone_numbers: tuple[str, ...]) -> bool:
    normalized = _normalize_phone(number)
    return any(normalized == _normalize_phone(official) for official in official_phone_numbers)


# 문자에 언급된 기관명과 실제 링크된 URL의 도메인, 그리고 본문에 적힌 전화번호가 그 기관의
# 공식 도메인/대표번호와 일치하는지 대조. 전화번호 대조는 URL 유무와 무관하게 동작하므로
# URL이 없는 문자(발신번호 사칭형 등)에서도 유일한 판정 신호가 될 수 있다.
def analyze_institution_match(text: str, traced_url: str | None = None) -> dict:
    no_institution_result = {
        "checked": False,
        "mismatch": False,
        "institution": None,
        "official_domains": [],
        "text_domain": None,
        "phone_checked": False,
        "phone_mismatch": False,
        "text_phone_numbers": [],
        "official_phone_numbers": [],
    }

    institution = _find_mentioned_institution(text)
    if institution is None:
        return no_institution_result

    text_phone_numbers = _extract_phone_numbers(text)
    phone_checked = bool(text_phone_numbers) and bool(institution.official_phone_numbers)
    phone_mismatch = phone_checked and not any(
        _matches_official_phone(number, institution.official_phone_numbers)
        for number in text_phone_numbers
    )
    if phone_mismatch:
        logger.warning(
            "[Institution] 기관명-대표번호 불일치 감지 - 기관: %s, 문자 내 번호: %s, 공식 대표번호: %s",
            institution.name,
            text_phone_numbers,
            institution.official_phone_numbers,
        )

    domain = _extract_domain(traced_url) if traced_url else None
    if domain is None:
        return {
            "checked": False,
            "mismatch": False,
            "institution": institution.name,
            "official_domains": list(institution.official_domains),
            "text_domain": None,
            "phone_checked": phone_checked,
            "phone_mismatch": phone_mismatch,
            "text_phone_numbers": text_phone_numbers,
            "official_phone_numbers": list(institution.official_phone_numbers),
        }

    is_official = _is_official_domain(domain, institution.official_domains)
    if not is_official:
        logger.warning(
            "[Institution] 기관명-도메인 불일치 감지 - 기관: %s, 문자 도메인: %s, 공식 도메인: %s",
            institution.name,
            domain,
            institution.official_domains,
        )

    return {
        "checked": True,
        "mismatch": not is_official,
        "institution": institution.name,
        "official_domains": list(institution.official_domains),
        "text_domain": domain,
        "phone_checked": phone_checked,
        "phone_mismatch": phone_mismatch,
        "text_phone_numbers": text_phone_numbers,
        "official_phone_numbers": list(institution.official_phone_numbers),
    }
