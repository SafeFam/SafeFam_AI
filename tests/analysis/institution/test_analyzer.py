from app.analysis.institution.analyzer import analyze_institution_match


def test_no_url_skips_check():
    result = analyze_institution_match("[국민은행] 계좌 확인 안내입니다.", traced_url=None)

    assert result["checked"] is False
    assert result["mismatch"] is False


def test_no_institution_mentioned_skips_check():
    result = analyze_institution_match(
        "택배가 도착했습니다", traced_url="https://random-domain.xyz"
    )

    assert result["checked"] is False
    assert result["mismatch"] is False


def test_mismatched_domain_is_flagged():
    result = analyze_institution_match(
        "[국민은행] 계좌 확인 안내입니다.", traced_url="https://kb-bank-security.xyz/login"
    )

    assert result["checked"] is True
    assert result["mismatch"] is True
    assert result["institution"] == "국민은행"
    assert result["official_domains"] == ["kbstar.com"]
    assert result["text_domain"] == "kb-bank-security.xyz"


def test_official_domain_is_not_flagged():
    result = analyze_institution_match(
        "[국민은행] 계좌 확인 안내입니다.", traced_url="https://obank.kbstar.com/login"
    )

    assert result["checked"] is True
    assert result["mismatch"] is False


def test_official_subdomain_is_recognized():
    """공식 도메인의 하위 도메인(예: 모바일/전용 서비스용)도 공식으로 인정해야 한다."""
    result = analyze_institution_match(
        "[신한은행] 인증 안내", traced_url="https://m.shinhan.com/verify"
    )

    assert result["checked"] is True
    assert result["mismatch"] is False


def test_institution_name_outside_bracket_tag_is_not_treated_as_sender():
    """
    "[롯데택배] ... 보내신 분: 농협하나로마트" 처럼 발신 주체가 아니라 상호명 등으로
    본문 중간에 기관명이 스쳐 지나가는 경우까지 사칭으로 오인해선 안 된다.
    실제 발신 주체는 대괄호 태그로 표기되는 SMS 관례를 따라 판단한다.
    """
    text = (
        "[롯데택배] 상품 배송 안내 보내신 분 : 농협하나로마트 "
        "실시간 배송현황 조회하기 https://www.lotteglogis.com/12345"
    )
    result = analyze_institution_match(text, traced_url="https://www.lotteglogis.com/12345")

    assert result["checked"] is False


def test_lookalike_domain_is_not_falsely_treated_as_official():
    """공식 도메인 문자열을 포함하지만 실제로는 다른 도메인인 경우(예: shinhan.com.evil.net)를
    하위 도메인으로 오인해선 안 된다."""
    result = analyze_institution_match(
        "[신한은행] 인증 안내", traced_url="https://shinhan.com.evil.net/verify"
    )

    assert result["checked"] is True
    assert result["mismatch"] is True
