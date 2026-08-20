import numpy as np
import pytest

from app.analysis.text.preprocessing import (
    AMOUNT_PATTERN,
    STRUCT_FEATURE_NAMES,
    contains_account_number,
    extract_struct_feature_matrix,
    extract_struct_features,
    mask_pii,
    normalize_text,
)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (
            "010-1234-5678로 연락주세요",
            "[PHONE]로 연락주세요",
        ),
        (
            "test@example.com으로 보내주세요",
            "[EMAIL]으로 보내주세요",
        ),
        (
            "카드번호 1234-5678-9012-3456",
            "카드번호 [CARD]",
        ),
        (
            "주민번호 900101-1234567",
            "주민번호 [RRN]",
        ),
        (
            "계좌 123-456-789012로 입금",
            "계좌 [ACCOUNT]로 입금",
        ),
    ],
)
def test_mask_pii_replaces_personal_information(
    original: str,
    expected: str,
):
    assert mask_pii(original) == expected


def test_normalize_text_masks_url_and_amount():
    text = "http://bit.ly/fake 계좌로 500,000원 즉시 입금하세요"

    normalized = normalize_text(text)

    assert normalized == ("[URL] 계좌로 [AMOUNT] 즉시 입금하세요")


def test_normalize_text_collapses_whitespace():
    text = "  긴급\t\t확인\n\n필요  "

    normalized = normalize_text(text)

    assert normalized == "긴급 확인 필요"


def test_normalize_text_removes_format_artifacts():
    text = "■■ 긴급 == 확인 □□"

    normalized = normalize_text(text)

    assert normalized == "긴급 확인"


def test_normalize_text_does_not_mask_digits_inside_url():
    text = "확인 주소 https://example.com/01012345678"

    normalized = normalize_text(text)

    assert normalized == "확인 주소 [URL]"
    assert "[PHONE]" not in normalized


def test_normalize_text_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        normalize_text(None)  # type: ignore[arg-type]


def test_extract_struct_features_detects_expected_signals():
    text = "[Web발신] 010-1234-5678로 연락하세요. 500,000원 확인: http://bit.ly/fake"

    features = extract_struct_features(text)

    assert features == [
        1,  # has_url
        1,  # has_short_url
        1,  # has_phone
        1,  # has_amount
        1,  # has_web_tag
        0,  # is_long_text
    ]


def test_extract_struct_features_detects_long_text():
    text = "가" * 101

    features = extract_struct_features(text)

    assert features[-1] == 1


def test_extract_struct_features_accepts_precomputed_url_flag():
    features = extract_struct_features(
        "URL이 제거된 메시지",
        has_url=True,
    )

    assert features[0] == 1


def test_extract_struct_feature_matrix_returns_expected_shape():
    texts = [
        "일반 메시지",
        "http://bit.ly/fake 확인",
    ]

    matrix = extract_struct_feature_matrix(texts)

    assert matrix.shape == (2, len(STRUCT_FEATURE_NAMES))
    assert matrix.dtype == np.int8
    assert matrix[0, 0] == 0
    assert matrix[1, 0] == 1
    assert matrix[1, 1] == 1


def test_extract_struct_feature_matrix_uses_precomputed_url_flags():
    texts = [
        "URL이 제거된 첫 번째 메시지",
        "URL이 없는 두 번째 메시지",
    ]
    has_urls = [True, False]

    matrix = extract_struct_feature_matrix(texts, has_urls)

    assert matrix[:, 0].tolist() == [1, 0]


def test_extract_struct_feature_matrix_rejects_length_mismatch():
    with pytest.raises(
        ValueError,
        match="texts and has_urls must contain the same number",
    ):
        extract_struct_feature_matrix(
            ["첫 번째", "두 번째"],
            [True],
        )


def test_extract_struct_feature_matrix_handles_empty_input():
    matrix = extract_struct_feature_matrix([])

    assert matrix.shape == (0, len(STRUCT_FEATURE_NAMES))


def test_single_and_batch_url_features_use_the_same_signal():
    text = "https://example.com에서 확인하세요"

    single = extract_struct_features(text)
    batch = extract_struct_feature_matrix([text])[0].tolist()

    assert batch == single


def test_contains_account_number_excludes_phone_numbers() -> None:
    """전화번호로 완전히 읽히는 숫자는 계좌로 세지 않는다"""
    assert not contains_account_number("예약 문의 02-345-6789")
    assert not contains_account_number("무료거부080-870-1234")
    assert not contains_account_number("상담 010-1234-5678")


def test_contains_account_number_keeps_real_accounts() -> None:
    """계좌 형식은 은행마다 다르므로 앞자리로 거르지 않는다"""
    assert contains_account_number("신한 110-234-567890")
    assert contains_account_number("1002-345-678901로 입금")


def test_masking_is_unchanged_by_the_account_check() -> None:
    """마스킹 경로는 그대로여야 fingerprint가 유지된다"""
    assert normalize_text("예약 문의 02-345-6789") == normalize_text(
        "예약 문의 02-345-6789"
    )
    assert "[PHONE]" in normalize_text("예약 문의 02-345-6789")
    assert "[ACCOUNT]" in normalize_text("신한 110-234-567890")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("88,435원", "88,435원"),
        ("113만 원", "113만 원"),
        ("3만원", "3만원"),
        ("5천원", "5천원"),
        ("1억원", "1억원"),
        ("1억 2천만원", "1억 2천만원"),
        ("3만 5천원", "3만 5천원"),
    ],
)
def test_amount_pattern_covers_korean_units(
    text: str,
    expected: str,
) -> None:
    """한글 단위 금액을 놓치면 금액 변형이 중복 제거를 통과한다"""
    match = AMOUNT_PATTERN.search(text)

    assert match is not None
    assert match.group() == expected


@pytest.mark.parametrize(
    "text",
    ["가격 5000\n\n원래는", "25일은 원장님 휴진"],
)
def test_amount_pattern_does_not_reach_across_lines(text: str) -> None:
    """공백을 한 칸으로 제한해 줄바꿈 너머의 '원'을 삼키지 않는다"""
    assert AMOUNT_PATTERN.search(text) is None


def test_amount_variants_share_one_fingerprint() -> None:
    """금액만 다른 같은 문장은 한 건으로 합쳐져야 한다"""
    masked = {
        normalize_text(f"예상환급액: {amount}")
        for amount in ("88,435원", "113만 원", "12만 원", "89만원")
    }

    assert len(masked) == 1
