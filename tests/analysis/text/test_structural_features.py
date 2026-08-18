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
    assert STACKING_STRUCTURAL_FEATURE_NAMES[-2:] == (
        "has_ad_disclosure",
        "has_opt_out",
    )
