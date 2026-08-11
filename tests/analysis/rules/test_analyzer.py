from app.analysis.rules.analyzer import analyze_text_with_rules


def test_clean_casual_message_scores_zero():
    result = analyze_text_with_rules("엄마 오늘 저녁 메뉴 뭐야?")

    assert result["rule_score"] == 0
    assert result["matched_rules"] == []
    assert result["has_malicious_domain_pattern"] is False


def test_malicious_domain_pattern_alone_maxes_out_score():
    """로컬 가드 도메인 룰(.ru, testsafebrowsing)이 매치되면 단독으로 만점(100)이어야 한다 (기존 동작 유지)."""
    result = analyze_text_with_rules(
        "확인하세요", traced_url="https://malicious.ru/phish"
    )

    assert result["rule_score"] == 100
    assert result["has_malicious_domain_pattern"] is True


def test_testsafebrowsing_domain_hint_is_detected():
    result = analyze_text_with_rules(
        "링크 확인", traced_url="https://testsafebrowsing.appspot.com/s/malware.html"
    )

    assert result["has_malicious_domain_pattern"] is True
    assert result["rule_score"] == 100


def test_account_number_pattern_is_detected():
    result = analyze_text_with_rules("입금 계좌 110-1234-567890 으로 보내주세요")

    assert result["rule_score"] >= 30
    assert any("계좌번호" in r for r in result["matched_rules"])


def test_card_number_pattern_is_detected():
    result = analyze_text_with_rules("카드번호 1234-5678-9012-3456 을 입력해주세요")

    assert result["rule_score"] >= 30
    assert any("카드번호" in r for r in result["matched_rules"])


def test_institution_mention_alone_is_a_weak_signal():
    """기관명 언급만으로는 낮은 점수만 부여되어야 한다 (단순 언급은 약한 신호)."""
    result = analyze_text_with_rules("국민은행 앞에서 3시에 만나자")

    assert 0 < result["rule_score"] <= 15
    assert any("금융기관" in r for r in result["matched_rules"])


def test_urgency_keyword_categories_are_capped():
    """긴급 키워드는 카테고리당 집계되고 상한(30점)을 넘지 않아야 한다."""
    text = "계좌 정지 예정이니 압류 전에 연체금을 즉시 확인 후 카드 정지를 해제하세요"
    result = analyze_text_with_rules(text)

    assert result["rule_score"] <= 30
    assert result["rule_score"] > 0


def test_combined_signals_produce_high_score_for_realistic_smishing_text():
    """
    기관 사칭 + 계좌번호 + 긴급 키워드가 결합된 전형적 스미싱 문자는
    여러 신호가 합산되어 높은 점수가 나와야 한다.
    """
    text = "[국민은행] 고객님 계좌가 명의도용으로 계좌 정지 처리되었습니다. 확인 후 110-1234-567890으로 즉시 확인 바랍니다"
    result = analyze_text_with_rules(text)

    assert result["rule_score"] >= 60
    assert len(result["matched_rules"]) >= 3


def test_rule_score_never_exceeds_100():
    text = (
        "[국민은행] 계좌 정지 명의도용 압류 연체 카드 정지 부정 사용 지금 바로 확인 "
        "계좌번호 110-1234-567890 카드번호 1234-5678-9012-3456"
    )
    result = analyze_text_with_rules(text, traced_url="https://scam.ru/phish")

    assert result["rule_score"] == 100


def test_masked_account_token_is_detected():
    """Spring이 마스킹한 [ACCOUNT] 토큰도 계좌번호로 탐지되어야 한다."""
    result = analyze_text_with_rules("입금 계좌 [ACCOUNT] 으로 보내주세요")

    assert result["rule_score"] >= 30
    assert any("계좌번호" in r for r in result["matched_rules"])


def test_masked_card_token_is_detected():
    """Spring이 마스킹한 [CARD] 토큰도 카드번호로 탐지되어야 한다."""
    result = analyze_text_with_rules("카드번호 [CARD] 을 입력해주세요")

    assert result["rule_score"] >= 30
    assert any("카드번호" in r for r in result["matched_rules"])


def test_institution_domain_mismatch_adds_score_and_is_exposed_structurally():
    """
    문자에 언급된 기관명과 실제 링크된 도메인이 그 기관의 공식 도메인이 아니면,
    GSB/VT 블랙리스트 등재 여부와 무관하게 규칙 점수에 반영되고 구조화된 정보로 노출돼야 한다.
    """
    result = analyze_text_with_rules(
        "[국민은행] 계좌 확인 안내입니다.", traced_url="https://kb-bank-security.xyz/login"
    )

    assert result["rule_score"] >= 60  # 기관명 언급(15) + 도메인 불일치(50)
    assert any("도메인 불일치" in r for r in result["matched_rules"])
    assert result["institution_match"]["mismatch"] is True
    assert result["institution_match"]["institution"] == "국민은행"


def test_institution_official_domain_does_not_add_mismatch_score():
    result = analyze_text_with_rules(
        "[국민은행] 계좌 확인 안내입니다.", traced_url="https://obank.kbstar.com/login"
    )

    assert result["institution_match"]["mismatch"] is False
    assert not any("도메인 불일치" in r for r in result["matched_rules"])
