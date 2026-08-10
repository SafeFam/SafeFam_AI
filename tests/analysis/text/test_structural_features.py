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
