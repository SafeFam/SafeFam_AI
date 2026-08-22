from app.analysis.evidence import build_evidence
from app.analysis.schemas import EvidenceCategory


def test_no_signals_produces_no_evidence():
    items = build_evidence(rule_analysis=None, url_analysis=None, text_analysis=None)

    assert items == []


def test_institution_mismatch_produces_impersonation_card_with_batchim_josa():
    items = build_evidence(
        rule_analysis={
            "institution_match": {
                "mismatch": True,
                "institution": "국민은행",
            }
        },
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.INSTITUTION_IMPERSONATION
    assert items[0].title == "기관 사칭"
    assert items[0].description == "국민은행을 언급했지만 공식 도메인이 아닙니다."


def test_institution_mismatch_uses_reul_when_no_batchim():
    items = build_evidence(
        rule_analysis={
            "institution_match": {
                "mismatch": True,
                "institution": "카카오뱅크",
            }
        },
        url_analysis=None,
        text_analysis=None,
    )

    assert items[0].description == "카카오뱅크를 언급했지만 공식 도메인이 아닙니다."


def test_institution_match_without_mismatch_produces_no_card():
    items = build_evidence(
        rule_analysis={"institution_match": {"mismatch": False, "checked": True}},
        url_analysis=None,
        text_analysis=None,
    )

    assert items == []


def test_institution_phone_mismatch_produces_impersonation_card():
    items = build_evidence(
        rule_analysis={
            "institution_match": {
                "mismatch": False,
                "phone_mismatch": True,
                "institution": "국민은행",
            }
        },
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.INSTITUTION_IMPERSONATION
    assert items[0].description == (
        "국민은행을 언급했지만 문자 속 연락처가 공식 대표번호가 아닙니다."
    )


def test_institution_domain_and_phone_mismatch_produce_two_cards():
    items = build_evidence(
        rule_analysis={
            "institution_match": {
                "mismatch": True,
                "phone_mismatch": True,
                "institution": "국민은행",
            }
        },
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 2
    assert all(
        item.category == EvidenceCategory.INSTITUTION_IMPERSONATION for item in items
    )


def test_account_or_card_pattern_produces_personal_info_card():
    items = build_evidence(
        rule_analysis={"has_account_or_card_pattern": True},
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.PERSONAL_INFO_REQUEST


def test_malicious_shortened_url_mentions_short_url_in_description():
    items = build_evidence(
        rule_analysis=None,
        url_analysis={"is_url_malicious": True, "is_shortened": True},
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.DANGEROUS_URL
    assert "단축 URL" in items[0].description


def test_malicious_non_shortened_url_produces_generic_description():
    items = build_evidence(
        rule_analysis=None,
        url_analysis={"is_url_malicious": True, "is_shortened": False},
        text_analysis=None,
    )

    assert "단축 URL" not in items[0].description


def test_local_guard_domain_pattern_alone_produces_dangerous_url_card():
    items = build_evidence(
        rule_analysis={"has_malicious_domain_pattern": True},
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.DANGEROUS_URL


def test_urgency_categories_produce_pressure_card():
    items = build_evidence(
        rule_analysis={"urgency_categories": ["긴급 확인 요구"]},
        url_analysis=None,
        text_analysis=None,
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.URGENCY_PRESSURE


def test_dangerous_text_grade_exposes_llm_reason_verbatim():
    items = build_evidence(
        rule_analysis=None,
        url_analysis=None,
        text_analysis={
            "result": {
                "grade": "DANGEROUS",
                "reason": "지인을 사칭한 금전 요구 문맥 감지",
            }
        },
    )

    assert len(items) == 1
    assert items[0].category == EvidenceCategory.AI_JUDGMENT
    assert items[0].description == "지인을 사칭한 금전 요구 문맥 감지"


def test_safe_text_grade_produces_no_ai_judgment_card():
    items = build_evidence(
        rule_analysis=None,
        url_analysis=None,
        text_analysis={"result": {"grade": "SAFE", "reason": "위험 요소 없음"}},
    )

    assert items == []


def test_all_signals_combine_into_four_cards_in_order():
    items = build_evidence(
        rule_analysis={
            "institution_match": {"mismatch": True, "institution": "신한은행"},
            "has_account_or_card_pattern": True,
            "has_malicious_domain_pattern": True,
            "urgency_categories": ["긴급 확인 요구"],
        },
        url_analysis={"is_url_malicious": True, "is_shortened": True},
        text_analysis={
            "result": {"grade": "DANGEROUS", "reason": "명의도용 사칭 문맥 감지"}
        },
    )

    assert [item.category for item in items] == [
        EvidenceCategory.INSTITUTION_IMPERSONATION,
        EvidenceCategory.PERSONAL_INFO_REQUEST,
        EvidenceCategory.DANGEROUS_URL,
        EvidenceCategory.URGENCY_PRESSURE,
        EvidenceCategory.AI_JUDGMENT,
    ]
