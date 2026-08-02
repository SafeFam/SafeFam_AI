import pytest
from pydantic import ValidationError

from app.chat.schemas import AnalysisContext, ChatMessage, ChatRequest


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
    context = _analysis_context(indicators=[])
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


def test_analysis_context_rejects_blank_explanation():
    with pytest.raises(ValidationError):
        _analysis_context(explanation="   ")


def test_analysis_context_indicator_requires_type_and_description():
    with pytest.raises(ValidationError):
        _analysis_context(indicators=[{"type": "MALICIOUS_URL"}])


def test_chat_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ChatRequest(
            analysisContext=_analysis_context(),
            messages=[{"role": "user", "content": "hi"}],
            unexpected="not allowed",
        )
