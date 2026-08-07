"""Kiwi 기반 SMS tokenizer 단위 테스트"""

import joblib
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from data_science.SMSModel.tokenization import (
    MASK_TOKENS,
    SELECTED_POS_TAGS,
    kiwi_tokenize,
)
from data_science.SMSModel.tokenization.kiwi_tokenizer import (
    _base_pos_tag,
    _get_kiwi,
)


def test_tokenize_korean_phishing_message():
    tokens = kiwi_tokenize(
        "계좌가 정지되었으니 즉시 본인 인증을 진행하세요."
    )

    assert "계좌" in tokens
    assert "정지" in tokens
    assert "즉시" in tokens
    assert "인증" in tokens
    assert "진행" in tokens


def test_tokenize_normal_message():
    tokens = kiwi_tokenize(
        "오늘 저녁에 같이 식사할까요?"
    )

    assert "오늘" in tokens
    assert "저녁" in tokens
    assert "같이" in tokens
    assert "식사" in tokens


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_empty_or_whitespace_input_returns_empty_list(text):
    assert kiwi_tokenize(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "!!!",
        "...",
        "()[]{}",
        "😊🚨",
    ],
)
def test_special_character_only_input_returns_empty_list(text):
    assert kiwi_tokenize(text) == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_non_string_input_raises_type_error(value):
    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        kiwi_tokenize(value)


@pytest.mark.parametrize(
    "mask_token",
    sorted(MASK_TOKENS),
)
def test_mask_token_is_preserved(mask_token):
    tokens = kiwi_tokenize(
        f"확인이 필요합니다 {mask_token}"
    )

    assert tokens.count(mask_token) == 1


def test_multiple_mask_tokens_are_preserved_in_order():
    tokens = kiwi_tokenize(
        "[URL]에서 [ACCOUNT] 정보를 확인하세요."
    )

    url_index = tokens.index("[URL]")
    account_index = tokens.index("[ACCOUNT]")

    assert url_index < account_index


def test_particle_and_ending_are_removed():
    text = "고객님의 계좌가 정지되었습니다."
    result = kiwi_tokenize(text)

    raw_tokens = _get_kiwi().tokenize(
        text,
        normalize_coda=True,
    )
    excluded_forms = {
        token.form
        for token in raw_tokens
        if _base_pos_tag(token.tag).startswith(("J", "E"))
    }

    assert excluded_forms.isdisjoint(result)


def test_selected_pos_tags_are_explicitly_defined():
    assert SELECTED_POS_TAGS == {
        "NNG",
        "NNP",
        "NNB",
        "NR",
        "NP",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MM",
        "MAG",
        "MAJ",
        "XR",
        "SL",
    }


def test_repeated_calls_return_same_tokens():
    text = "긴급 대출 신청을 위해 본인 인증이 필요합니다."

    first = kiwi_tokenize(text)
    second = kiwi_tokenize(text)

    assert first == second


def test_kiwi_instance_is_reused():
    assert _get_kiwi() is _get_kiwi()


def test_irregular_pos_suffix_is_normalized():
    """Kiwi irregular-conjugation suffixes must not bypass POS filtering."""
    assert _base_pos_tag("VV-I") == "VV"
    assert _base_pos_tag("VA-R") == "VA"
    assert _base_pos_tag("NNG") == "NNG"


def test_tokenizer_integrates_with_count_vectorizer():
    """The public tokenizer must work as a scikit-learn callable."""
    vectorizer = CountVectorizer(
        tokenizer=kiwi_tokenize,
        token_pattern=None,
        lowercase=False,
    )

    matrix = vectorizer.fit_transform(
        [
            "계좌가 정지되었습니다 [URL]",
            "오늘 저녁 식사 약속입니다",
        ]
    )

    assert matrix.shape[0] == 2
    assert "[URL]" in vectorizer.vocabulary_
    assert "계좌" in vectorizer.vocabulary_


def test_vectorizer_with_tokenizer_survives_joblib_round_trip(tmp_path):
    """A trained vectorizer must remain usable after artifact serialization."""
    vectorizer = CountVectorizer(
        tokenizer=kiwi_tokenize,
        token_pattern=None,
        lowercase=False,
    )
    vectorizer.fit(["본인 인증이 필요합니다 [URL]"])
    artifact_path = tmp_path / "kiwi_vectorizer.pkl"

    joblib.dump(vectorizer, artifact_path)
    restored = joblib.load(artifact_path)

    original = vectorizer.transform(["본인 인증 [URL]"])
    reloaded = restored.transform(["본인 인증 [URL]"])
    assert (original != reloaded).nnz == 0
