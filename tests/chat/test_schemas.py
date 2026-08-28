import pytest
from pydantic import ValidationError

from app.chat.schemas import AnalysisContext, ChatMessage, ChatRequest
from app.core.config import settings


def _analysis_context(**overrides) -> AnalysisContext:
    defaults = {
        "riskScore": 90,
        "riskLevel": "HIGH",
        "category": "FINANCIAL_INSTITUTION",
        "explanation": "국민건강보험을 사칭한 스미싱 문자",
        "indicators": [
            {"type": "MALICIOUS_URL", "description": "악성 이력이 확인된 URL입니다."}
        ],
    }
    defaults.update(overrides)
    return AnalysisContext(**defaults)


def test_chat_request_accepts_valid_payload():
    request = ChatRequest(
        analysisContext=_analysis_context(),
        messages=[{"role": "user", "content": "이거 진짜인가요?"}],
    )

    assert request.analysisContext.riskScore == 90
    assert request.messages[0].role.value == "user"


def test_analysis_context_defaults_indicators_to_empty_list():
    context = AnalysisContext(
        riskScore=90,
        riskLevel="HIGH",
        category="FINANCIAL_INSTITUTION",
        explanation="국민건강보험을 사칭한 스미싱 문자",
    )
    assert context.indicators == []


def test_chat_request_rejects_empty_message_history():
    with pytest.raises(ValidationError):
        ChatRequest(analysisContext=_analysis_context(), messages=[])


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="hi")


def test_chat_message_rejects_blank_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_chat_message_accepts_content_at_max_length():
    message = ChatMessage(
        role="user", content="가" * settings.MAX_CHAT_CONTENT_LENGTH
    )

    assert len(message.content) == settings.MAX_CHAT_CONTENT_LENGTH


def test_chat_message_rejects_content_over_max_length():
    with pytest.raises(ValidationError):
        ChatMessage(
            role="user", content="가" * (settings.MAX_CHAT_CONTENT_LENGTH + 1)
        )


def test_chat_message_oversized_content_is_not_leaked_in_error():
    """hide_input_in_errors=True 로 챗 원문(PII)이 에러에 담기지 않아야 한다."""
    secret_marker = "01012345678-비밀번호"
    with pytest.raises(ValidationError) as exception_info:
        ChatMessage(
            role="user",
            content=secret_marker + "가" * settings.MAX_CHAT_CONTENT_LENGTH,
        )

    assert secret_marker not in str(exception_info.value)


def test_chat_request_accepts_messages_at_max_count():
    request = ChatRequest(
        analysisContext=_analysis_context(),
        messages=[
            {"role": "user", "content": "질문"}
            for _ in range(settings.MAX_CHAT_MESSAGES)
        ],
    )

    assert len(request.messages) == settings.MAX_CHAT_MESSAGES


def test_chat_request_rejects_messages_over_max_count():
    with pytest.raises(ValidationError):
        ChatRequest(
            analysisContext=_analysis_context(),
            messages=[
                {"role": "user", "content": "질문"}
                for _ in range(settings.MAX_CHAT_MESSAGES + 1)
            ],
        )


def test_analysis_context_rejects_blank_explanation():
    with pytest.raises(ValidationError):
        _analysis_context(explanation="   ")


def test_analysis_context_indicator_requires_type_and_description():
    with pytest.raises(ValidationError):
        _analysis_context(indicators=[{"type": "MALICIOUS_URL"}])
    with pytest.raises(ValidationError):
        _analysis_context(indicators=[{"description": "악성 이력이 확인된 URL입니다."}])


def test_chat_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(
            analysisContext=_analysis_context(),
            messages=[{"role": "user", "content": "hi"}],
            unexpected="not allowed",
        )
