"""Stacking 구조 특징 추출기 테스트"""

import numpy as np
import pytest

from app.analysis.text.structural_features import (
    STACKING_STRUCTURAL_FEATURE_NAMES,
    extract_stacking_structural_features,
    extract_stacking_structural_matrix,
)


def test_extracts_phishing_structural_features() -> None:
    text = (
        "[Web발신] 계좌가 정지될 예정입니다. "
        "즉시 110-1234-567890으로 100,000원 송금 후 "
        "https://bit.ly/example 링크를 클릭하세요."
    )

    result = extract_stacking_structural_features(text)
    features = dict(zip(result.names, result.values, strict=True))

    assert features["has_url"] == 1.0
    assert features["has_short_url"] == 1.0
    assert features["has_account"] == 1.0
    assert features["has_amount"] == 1.0
    assert features["has_web_tag"] == 1.0
    assert features["has_urgency"] == 1.0
    assert features["has_transfer_request"] == 1.0
    assert features["has_link_action"] == 1.0


def test_normal_message_has_no_risky_features() -> None:
    result = extract_stacking_structural_features(
        "엄마 오늘 저녁 메뉴 뭐야?"
    )

    assert np.array_equal(
        result.values,
        np.zeros(len(STACKING_STRUCTURAL_FEATURE_NAMES)),
    )


def test_returns_fixed_feature_order() -> None:
    result = extract_stacking_structural_features("테스트 메시지")

    assert result.names == STACKING_STRUCTURAL_FEATURE_NAMES
    assert len(result.values) == len(STACKING_STRUCTURAL_FEATURE_NAMES)


def test_builds_matrix_for_multiple_messages() -> None:
    matrix = extract_stacking_structural_matrix(
        [
            "일상 대화입니다.",
            "즉시 계좌로 송금하세요.",
        ]
    )

    assert matrix.shape == (
        2,
        len(STACKING_STRUCTURAL_FEATURE_NAMES),
    )


def test_empty_input_returns_empty_matrix() -> None:
    matrix = extract_stacking_structural_matrix([])

    assert matrix.shape == (
        0,
        len(STACKING_STRUCTURAL_FEATURE_NAMES),
    )


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        extract_stacking_structural_features(123)


def test_detects_legal_advertising_disclosure() -> None:
    """법정 광고 표기와 수신거부 안내를 특징으로 잡아내는지 검증"""
    result = extract_stacking_structural_features(
        "(광고)[나이키] 시즌 마감 세일 40% 할인 무료수신거부 080-123-4567"
    )
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_ad_disclosure"] == 1.0
    assert values["has_opt_out"] == 1.0


def test_detects_bracket_style_advertising_disclosure() -> None:
    """[광고] 형태와 공백이 섞인 표기도 인식해야 한다"""
    result = extract_stacking_structural_features(
        "[ 광고 ] 신규 회원 쿠폰 안내입니다. 수신 거부 080 1234 5678"
    )
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_ad_disclosure"] == 1.0
    assert values["has_opt_out"] == 1.0


def test_plain_notification_has_no_advertising_marks() -> None:
    """광고 표기가 없는 정상 알림은 두 특징 모두 0이어야 한다"""
    result = extract_stacking_structural_features(
        "[국민은행] 출금 50,000원 잔액 120,000원"
    )
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_ad_disclosure"] == 0.0
    assert values["has_opt_out"] == 0.0


def test_feature_matrix_width_matches_names() -> None:
    """특징을 추가해도 행렬 폭과 이름 개수가 일치해야 한다"""
    matrix = extract_stacking_structural_matrix(["테스트 문자", "두 번째 문자"])

    assert matrix.shape == (2, len(STACKING_STRUCTURAL_FEATURE_NAMES))


def test_new_features_are_appended_last() -> None:
    """열 순서 호환을 위해 새 특징은 항상 마지막에 있어야 한다"""
    # 콤마 누락으로 인접 문자열이 암묵적으로 이어붙으면 개수부터 어긋난다.
    assert len(STACKING_STRUCTURAL_FEATURE_NAMES) == 18
    # 광고 특징(#84)은 자리를 지켜야 기존 열 인덱스가 보존된다.
    assert STACKING_STRUCTURAL_FEATURE_NAMES[12:14] == (
        "has_ad_disclosure",
        "has_opt_out",
    )
    assert STACKING_STRUCTURAL_FEATURE_NAMES[-4:] == (
        "has_family_impersonation",
        "has_chatroom_invite",
        "has_job_offer_lure",
        "has_investment_lure",
    )


def test_detects_advertising_disclosure_after_web_prefix() -> None:
    """`[Web발신]` 접두어 다음 줄에서 시작하는 광고 표기도 인식해야 한다"""
    result = extract_stacking_structural_features(
        "[Web발신]\n광고 신상품 안내입니다.\n무료수신거부 080-123-4567"
    )
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_ad_disclosure"] == 1.0
    assert values["has_opt_out"] == 1.0


def test_advertising_disclosure_ignores_mid_line_mention() -> None:
    """문장 중간에 언급된 '광고'는 법정 표기로 보지 않는다"""
    result = extract_stacking_structural_features(
        "어제 본 광고 기억나? 그거 링크 좀 보내줘."
    )
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_ad_disclosure"] == 0.0


def test_opt_out_covers_free_variants() -> None:
    """무료수신거부·수신 거부·무료거부 표기를 모두 같은 특징으로 잡는다"""
    for text in ("무료수신거부", "수신 거부", "무료거부", "080-123-4567"):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_opt_out"] == 1.0, text


def test_normal_authentication_message_is_not_a_credential_request() -> None:
    """정상 인증 문자는 개인정보 요구로 잡히면 안 된다

    이 특징은 피싱 전용 신호인데, 단어 존재만 보던 시절에는 정상 인증 문자가
    전부 켜서 판별력이 사라졌다(#92).
    """
    for text in (
        "[Web발신][배달의민족] 인증번호 [198788]를 입력해주세요.",
        "[Web발신][토스] 인증번호 [830101] 입니다.",
        "[Web발신][KCB] 본인확인 인증번호는 685649입니다. 정확히 입력해주세요.",
        "[Web발신](광고)연세의원 국가건강검진 안내 방문 시 신분증 지참 부탁드립니다.",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_personal_info_request"] == 0.0, text


def test_handover_request_is_flagged() -> None:
    """제3자에게 넘기라는 요구는 잡아야 한다"""
    for text in (
        "인증번호를 알려주시면 처리해 드립니다",
        "보안카드 번호를 전송해 주세요",
        "주민등록번호를 회신 바랍니다",
        "계좌번호를 보내주세요",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_personal_info_request"] == 1.0, text


def test_phone_number_is_not_counted_as_an_account() -> None:
    """연락처가 계좌번호 신호를 켜면 안 된다

    소상공인 광고는 연락처와 수신거부 번호를 함께 싣는다. 그것이 계좌로
    잡히면 계좌번호가 정상 쪽 신호가 된다(#92).
    """
    for text in (
        "예약 문의 02-345-6789",
        "무료거부080-870-1234",
        "상담문의 010-1234-5678",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_account"] == 0.0, text


def test_real_account_number_is_still_detected() -> None:
    """전화번호를 걸러내면서 실제 계좌번호는 계속 잡아야 한다"""
    for text in (
        "신한 110-234-567890으로 입금해 주세요",
        "계좌 1002-345-678901",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_account"] == 1.0, text


def test_privacy_notices_are_not_credential_requests() -> None:
    """안내문은 요구가 아니다

    어간만 보면 "개인정보 제공에 동의"나 "개인정보를 알려드립니다" 같은
    문장이 걸린다. 둘 다 받는 사람에게 자격증명을 넘기라는 말이 아니다.
    """
    for text in (
        "개인정보 제공에 동의합니다",
        "개인정보를 알려드립니다",
        "개인정보 제공 동의 안내입니다",
        "인증번호 안내를 보내드렸습니다",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_personal_info_request"] == 0.0, text


def test_imperative_handover_forms_are_still_flagged() -> None:
    """요구 어미가 붙은 형태는 계속 잡아야 한다"""
    for text in (
        "인증번호 알려줘",
        "계좌번호 회신요망",
        "비밀번호를 알려주십시오",
        "보안카드 번호를 전송 부탁드립니다",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_personal_info_request"] == 1.0, text


def test_family_impersonation_is_flagged() -> None:
    """가족을 사칭하며 연락 수단이 바뀌었다고 둘러대는 형태

    문체가 일상 대화와 같아 어휘만으로는 걸러지지 않는다. 친족 호칭과
    폰 고장/임시번호/대리 송금이 함께 오는 구조로 잡는다.
    """
    for text in (
        "엄마 나 폰 수리 맡겼는데 주민등록증 사진 보내줘",
        "아빠 휴대폰이 고장 나서 임시번호야. 급한 병원비 70만원만 보내줘.",
        "장모님 저 대신 먼저 보내주시면 안되나해서요 401만 원이에요",
        "형수님 공인인증서가 만료가 되어서 이체가 안되는데 부탁드려도 될까요",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_family_impersonation"] == 1.0, text


def test_ordinary_family_talk_is_not_impersonation() -> None:
    """호칭만으로 걸리면 일상 대화가 통째로 오탐이 된다"""
    for text in (
        "엄마 오늘 저녁 뭐야?",
        "아빠 주말에 등산 가신대",
        "누나 생일 선물 뭐 살까",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_family_impersonation"] == 0.0, text


def test_chatroom_invite_is_flagged() -> None:
    """통신사 문자 밖 대화방으로 넘기려는 유도"""
    for text in (
        "오늘 급등 종목 무료 공개. 오픈채팅 입장 후 투자금 입금 바랍니다.",
        "즉시 저희 방에 놀러오세요. 아래 링크 누르시고 밴드에 입장하시면",
        "VIP 정보방 입장 코드: 7771",
        "채용 담당자 추가 후 상세 정보 확인 가능 .카톡ID: ******",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_chatroom_invite"] == 1.0, text


def test_plus_friend_registration_is_not_a_chatroom_invite() -> None:
    """플러스친구 등록은 합법 광고의 표준 문구다

    이 문구를 대화방 유도로 보면 정상 광고 문자가 통째로 걸린다.
    """
    text = (
        "[Web발신] (광고)경희한의원 "
        "카카오톡 플러스 친구 등록 후 추가 혜택 받으세요!!"
    )
    result = extract_stacking_structural_features(text)
    values = dict(zip(result.names, result.values, strict=True))

    assert values["has_chatroom_invite"] == 0.0


def test_job_offer_lure_is_flagged() -> None:
    """손쉬운 고수익을 내세운 채용 및 부업 유인"""
    for text in (
        "리뷰 작성 재택업무 일급 30만원. 업무 시작 전 예치금이 필요합니다.",
        "온라인 아르바이트 모집 중, 시급: 10,000, 급여 실시간 지급",
        "아르바이트, 풀타임 모두 가능하며 하루 20-112만 원의 추가 수입",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_job_offer_lure"] == 1.0, text


def test_investment_lure_is_flagged() -> None:
    """종목 추천이나 수익률을 내세운 투자 유인"""
    for text in (
        "우수 고객 무료 추천 종목 서비스 당첨되셨습니다.",
        "매주 수익률이 최대 33.59%에 달하는 우량주를 준비했어요. "
        "채팅방에 참여하시면 즉시 5000원을 드리고요",
        "[단독입수] 3일 만에 300% 폭등할 하반기 주도주 무료 공개!",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert values["has_investment_lure"] == 1.0, text


def test_casual_conversation_triggers_no_lure_feature() -> None:
    """일상 대화에서는 네 유인 특징이 하나도 켜지지 않아야 한다"""
    lure_names = (
        "has_family_impersonation",
        "has_chatroom_invite",
        "has_job_offer_lure",
        "has_investment_lure",
    )
    for text in (
        "동탄신도시도 아니고 몽탄신도시는 도대체 어디야?ㅋㅋ",
        "ㅎㅎ나도 그래. 어떤 경기를 볼 수 있을지 너무 기대된다.",
        "배달기사입니다. 문앞에 놓고갑니다. 맛있게 드세요.",
    ):
        result = extract_stacking_structural_features(text)
        values = dict(zip(result.names, result.values, strict=True))

        assert not any(values[name] for name in lure_names), text
