import pytest
from pydantic import ValidationError

from app.chat.schemas import AnalysisContext, ChatMessage, ChatRequest


def _analysis_context(**overrides) -> AnalysisContext:
    defaults = {
        "riskScore": 90,
        "riskGrade": "HIGH",
        "phishingType": "기관 사칭형",
        "summary": "국민건강보험을 사칭한 스미싱 문자",
    }
    defaults.update(overrides)
    return AnalysisContext(**defaults)


def test_chat_request_accepts_valid_payload():
    request = ChatRequest(
        analysisContext=_analysis_context(),
        indicators=["국민건강보험 언급", "즉시 확인 유도"],
        messages=[{"role": "user", "content": "이거 진짜인가요?"}],
    )

    assert request.analysisContext.riskScore == 90
    assert request.messages[0].role.value == "user"


def test_chat_request_defaults_indicators_to_empty_list():
    request = ChatRequest(
        analysisContext=_analysis_context(),
        messages=[{"role": "user", "content": "질문입니다"}],
    )

    assert request.indicators == []


def test_chat_request_rejects_empty_message_history():
    with pytest.raises(ValidationError):
        ChatRequest(analysisContext=_analysis_context(), indicators=[], messages=[])


def test_chat_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="hi")


def test_chat_message_rejects_blank_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="   ")


def test_analysis_context_rejects_blank_summary():
    with pytest.raises(ValidationError):
        _analysis_context(summary="   ")


def test_analysis_context_phishing_type_is_optional():
    context = _analysis_context(phishingType=None)
    assert context.phishingType is None


def test_chat_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(
            analysisContext=_analysis_context(),
            messages=[{"role": "user", "content": "hi"}],
            unexpected="not allowed",
        )
