import pytest

from app.analysis.phishing_type import PhishingType, classify_phishing_type


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("은행 계좌가 정지되었습니다", PhishingType.FINANCIAL_INSTITUTION),
        ("검찰 수사관입니다", PhishingType.GOVERNMENT_AGENCY),
        ("저금리 대환 대출 안내", PhishingType.LOAN),
        ("재택근무 고수익 알바 모집", PhishingType.JOB),
        ("택배 배송 주소지를 확인하세요", PhishingType.DELIVERY),
        ("엄마 휴대폰 고장났어 카톡 추가해", PhishingType.MESSENGER),
        ("오늘 저녁에 만나요", PhishingType.OTHER),
    ],
)
def test_classifies_supported_phishing_types(
    content: str,
    expected: PhishingType,
) -> None:
    assert classify_phishing_type(content) == expected


def test_selects_category_with_most_keyword_matches() -> None:
    result = classify_phishing_type("은행 안내입니다 택배 배송 운송장을 확인하세요")

    assert result == PhishingType.DELIVERY


def test_uses_contract_order_to_break_ties() -> None:
    result = classify_phishing_type("은행 계좌 관련 검찰 경찰 안내")

    assert result == PhishingType.FINANCIAL_INSTITUTION


def test_includes_sender_in_classification() -> None:
    result = classify_phishing_type("인증 안내", sender="국세청")

    assert result == PhishingType.GOVERNMENT_AGENCY
