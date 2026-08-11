"""Gemini validation 캐시 입력 검증 테스트."""

import pytest

from data_science.SMSModel.run_hybrid_threshold_selection import (
    _is_available_gemini_result,
)


@pytest.mark.parametrize("score", [True, False])
def test_rejects_boolean_gemini_scores(score: bool) -> None:
    assert _is_available_gemini_result(
        score=score,
        grade="DANGEROUS",
        error_message=None,
    ) is False


def test_accepts_integer_gemini_score() -> None:
    assert _is_available_gemini_result(
        score=85,
        grade="DANGEROUS",
        error_message=None,
    ) is True
