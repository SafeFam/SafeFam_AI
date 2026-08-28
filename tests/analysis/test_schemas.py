import pytest
from pydantic import ValidationError

from app.analysis.schemas import SmishingAnalysisRequest
from app.core.config import settings


def test_accepts_valid_text():
    request = SmishingAnalysisRequest(text="[국민은행] 계좌가 정지되었습니다.")

    assert request.text == "[국민은행] 계좌가 정지되었습니다."


@pytest.mark.parametrize("text", ["", " ", "   ", "\t", "\n"])
def test_rejects_blank_text(text: str):
    with pytest.raises(ValidationError):
        SmishingAnalysisRequest(text=text)


def test_accepts_text_at_max_length():
    request = SmishingAnalysisRequest(
        text="가" * settings.MAX_ANALYSIS_CONTENT_LENGTH
    )

    assert len(request.text) == settings.MAX_ANALYSIS_CONTENT_LENGTH


def test_rejects_text_over_max_length():
    with pytest.raises(ValidationError):
        SmishingAnalysisRequest(
            text="가" * (settings.MAX_ANALYSIS_CONTENT_LENGTH + 1)
        )


def test_oversized_text_is_not_leaked_in_validation_error():
    """hide_input_in_errors=True 로 검증 실패 시 원문(PII)이 에러에 담기지 않아야 한다."""
    secret_marker = "01012345678-비밀번호-보이스피싱"
    with pytest.raises(ValidationError) as exception_info:
        SmishingAnalysisRequest(
            text=secret_marker + "가" * settings.MAX_ANALYSIS_CONTENT_LENGTH
        )

    assert secret_marker not in str(exception_info.value)
