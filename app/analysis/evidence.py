"""분석 결과를 사용자 언어의 증거 카드로 변환 (issue #64 v1)

v1 범위: 이미 구조화된 규칙/URL/기관대조 신호만 카테고리별 카드로 매핑하고,
LLM 판단은 재설계 없이 단일 AI_JUDGMENT 카드로 원문(reason)을 그대로 노출한다.
"""

from app.analysis.schemas import EvidenceCategory, EvidenceItem

_CATEGORY_TITLES: dict[EvidenceCategory, str] = {
    EvidenceCategory.INSTITUTION_IMPERSONATION: "기관 사칭",
    EvidenceCategory.PERSONAL_INFO_REQUEST: "개인정보 요구",
    EvidenceCategory.DANGEROUS_URL: "위험 URL",
    EvidenceCategory.URGENCY_PRESSURE: "행동 압박",
    EvidenceCategory.AI_JUDGMENT: "AI 판단",
}

# LLM이 위험하다고 판단한 경우에만 AI_JUDGMENT 카드를 노출 (안전 판정에는 불필요)
_AI_JUDGMENT_GRADES = ("SUSPICIOUS", "DANGEROUS")


def _josa_eul_reul(word: str) -> str:
    """word의 마지막 글자 받침 유무에 따라 '을'/'를'을 반환 (한글이 아니면 '를')."""
    if not word:
        return "를"
    code = ord(word[-1]) - 0xAC00
    if 0 <= code <= 11171:
        return "을" if code % 28 != 0 else "를"
    return "를"


def _item(category: EvidenceCategory, description: str) -> EvidenceItem:
    return EvidenceItem(
        category=category,
        title=_CATEGORY_TITLES[category],
        description=description,
    )


def build_evidence(
    *,
    rule_analysis: dict | None,
    url_analysis: dict | None,
    text_analysis: dict | None,
) -> list[EvidenceItem]:
    """규칙/URL/기관대조/LLM 판단 결과를 증거 카드 목록으로 변환한다."""
    rule_analysis = rule_analysis or {}
    url_analysis = url_analysis or {}
    text_analysis = text_analysis or {}

    items: list[EvidenceItem] = []

    institution_match = rule_analysis.get("institution_match") or {}
    if institution_match.get("mismatch"):
        institution = institution_match.get("institution") or "해당 기관"
        items.append(
            _item(
                EvidenceCategory.INSTITUTION_IMPERSONATION,
                f"{institution}{_josa_eul_reul(institution)} 언급했지만 "
                "공식 도메인이 아닙니다.",
            )
        )

    if rule_analysis.get("has_account_or_card_pattern"):
        items.append(
            _item(
                EvidenceCategory.PERSONAL_INFO_REQUEST,
                "계좌 또는 카드 정보로 보이는 내용이 포함되어 있습니다.",
            )
        )

    if bool(url_analysis.get("is_url_malicious")) or bool(
        rule_analysis.get("has_malicious_domain_pattern")
    ):
        description = (
            "단축 URL의 최종 목적지가 위험한 것으로 확인됐습니다."
            if url_analysis.get("is_shortened")
            else "문자에 포함된 링크가 위험한 것으로 확인됐습니다."
        )
        items.append(_item(EvidenceCategory.DANGEROUS_URL, description))

    if rule_analysis.get("urgency_categories"):
        items.append(
            _item(
                EvidenceCategory.URGENCY_PRESSURE,
                '"즉시", "오늘까지" 등 빠른 판단을 재촉하는 표현이 있습니다.',
            )
        )

    text_result = text_analysis.get("result") or {}
    reason = text_result.get("reason")
    if text_result.get("grade") in _AI_JUDGMENT_GRADES and reason:
        items.append(_item(EvidenceCategory.AI_JUDGMENT, reason))

    return items
