from app.analysis.institution.registry import OFFICIAL_INSTITUTIONS
from app.analysis.rules.analyzer import FINANCIAL_INSTITUTIONS


def test_every_scored_institution_has_an_official_domain_entry():
    """FINANCIAL_INSTITUTIONS의 언급 가점(+15)만 받고 OFFICIAL_INSTITUTIONS의 도메인
    위변조 검사(+50, 단독으로 HIGH급)는 못 받는 기관이 생기지 않도록 막는다. 두 목록이
    벌어지면 그 기관을 사칭한 문자는 가장 강한 시그널 없이 조용히 약하게만 잡힌다."""
    known_aliases = {
        alias
        for institution in OFFICIAL_INSTITUTIONS
        for alias in institution.aliases
    }

    missing = [name for name in FINANCIAL_INSTITUTIONS if name not in known_aliases]

    assert not missing, (
        f"OFFICIAL_INSTITUTIONS에 도메인 등록이 안 된 기관: {missing}"
    )
