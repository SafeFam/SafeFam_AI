from app.chat.prompts import build_system_prompt
from app.chat.schemas import AnalysisContext


def test_build_system_prompt_includes_context_fields():
    context = AnalysisContext(
        riskScore=90,
        riskLevel="HIGH",
        category="FINANCIAL_INSTITUTION",
        explanation="국민건강보험을 사칭한 스미싱 문자",
        indicators=[
            {"type": "MALICIOUS_URL", "description": "악성 이력이 확인된 URL"},
            {"type": "URGENCY_KEYWORD", "description": "즉시 확인 유도 문구"},
        ],
    )

    prompt = build_system_prompt(context)

    assert "90/100" in prompt
    assert "HIGH" in prompt
    assert "RiskGrade" not in prompt
    assert "FINANCIAL_INSTITUTION" in prompt
    assert "국민건강보험을 사칭한 스미싱 문자" in prompt
    assert "MALICIOUS_URL: 악성 이력이 확인된 URL" in prompt
    assert "URGENCY_KEYWORD: 즉시 확인 유도 문구" in prompt


def test_build_system_prompt_handles_empty_indicators():
    context = AnalysisContext(
        riskScore=10,
        riskLevel="LOW",
        category="ETC",
        explanation="일상적인 대화",
    )

    prompt = build_system_prompt(context)

    assert "탐지 근거: 없음" in prompt
