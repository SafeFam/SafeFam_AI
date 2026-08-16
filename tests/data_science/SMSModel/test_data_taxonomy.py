"""SMS 보조 라벨 taxonomy 테스트"""

from __future__ import annotations

import pytest

from data_science.SMSModel.data_quality import (
    NORMAL_MESSAGE_TYPES,
    PHISHING_MESSAGE_TYPES,
    is_allowed_label_type_pair,
    normalize_legacy_message_type,
)


def test_normal_and_phishing_types_are_disjoint() -> None:
    assert NORMAL_MESSAGE_TYPES.isdisjoint(
        PHISHING_MESSAGE_TYPES
    )


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("투자리딩방사칭형", "투자리딩방사기"),
        ("투자_리딩방", "투자리딩방사기"),
        ("대출_사기", "대출사기"),
        ("중고거래_사기", "중고거래사기"),
        ("택배사칭", "택배배송사칭"),
    ],
)
def test_normalizes_legacy_types(
    legacy: str,
    expected: str,
) -> None:
    assert normalize_legacy_message_type(legacy) == expected


@pytest.mark.parametrize(
    ("label", "message_type"),
    [
        ("normal", "일상대화"),
        ("normal", "정상카드결제알림"),
        ("phishing", "택배배송사칭"),
        ("phishing", "투자리딩방사기"),
    ],
)
def test_accepts_valid_label_type_pairs(
    label: str,
    message_type: str,
) -> None:
    assert is_allowed_label_type_pair(
        label,
        message_type,
    )


@pytest.mark.parametrize(
    ("label", "message_type"),
    [
        ("normal", "택배배송사칭"),
        ("phishing", "정상카드결제알림"),
        ("unknown", "기타피싱"),
    ],
)
def test_rejects_invalid_label_type_pairs(
    label: str,
    message_type: str,
) -> None:
    assert not is_allowed_label_type_pair(
        label,
        message_type,
    )