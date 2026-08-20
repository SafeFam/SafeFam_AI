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


def test_advertising_features_are_appended_last() -> None:
    """열 순서 호환을 위해 새 특징은 항상 마지막에 있어야 한다"""
    # 콤마 누락으로 인접 문자열이 암묵적으로 이어붙으면 개수부터 어긋난다.
    assert len(STACKING_STRUCTURAL_FEATURE_NAMES) == 14
    assert STACKING_STRUCTURAL_FEATURE_NAMES[-2:] == (
        "has_ad_disclosure",
        "has_opt_out",
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
