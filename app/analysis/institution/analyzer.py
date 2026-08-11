import logging
import re
from urllib.parse import urlparse

from app.analysis.institution.registry import OFFICIAL_INSTITUTIONS, OfficialInstitution

logger = logging.getLogger(__name__)

# GSB/VT 블랙리스트는 신고 이력이 쌓여야 걸리므로 아직 신고되지 않은 신규 피싱 도메인은
# 놓칠 수 있다. 이 검사는 "그 기관의 공식 도메인이 아니다"라는 사실만으로 신고 이력과
# 무관하게 즉시 잡아낼 수 있어 단독으로도 강한 신호로 취급한다.
INSTITUTION_MISMATCH_SCORE = 50

# SMS 발신 표기 관례상 발신 주체는 "[국민은행]"처럼 대괄호 태그로 표기되는 경우가 대부분이다.
# 본문 전체에서 기관명을 찾으면 "보내신 분: 농협하나로마트" 같은, 발신 주체가 아니라 단순히
# 상호명 등으로 스쳐 지나가는 언급까지 사칭으로 오인해 정상 문자(택배 발신자명 등)를 오탐하므로,
# 실제 발신 주체를 나타내는 대괄호 태그 안으로 범위를 좁힌다.
_RE_BRACKET_TAG = re.compile(r"\[([^\[\]]+)\]")


def _extract_domain(url: str) -> str | None:
    hostname = urlparse(url).hostname
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


# 문자에 언급된 기관명과 실제 링크된 URL의 도메인이 그 기관의 공식 도메인과 일치하는지 대조.
# 기관 언급이 없거나 URL이 없으면 대조할 대상이 없으므로 checked=False로 스킵.
def analyze_institution_match(text: str, traced_url: str | None = None) -> dict:
    empty_result = {
        "checked": False,
        "mismatch": False,
        "institution": None,
        "official_domains": [],
        "text_domain": None,
    }

    if not traced_url:
        return empty_result

    domain = _extract_domain(traced_url)
    if not domain:
        return empty_result

    institution = _find_mentioned_institution(text)
    if institution is None:
        return empty_result

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
    }
