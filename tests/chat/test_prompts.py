from app.chat.prompts import build_system_prompt
from app.chat.schemas import AnalysisContext


def test_build_system_prompt_includes_context_fields():
    context = AnalysisContext(
        riskScore=90,
        riskGrade="HIGH",
        phishingType="기관 사칭형",
        summary="국민건강보험을 사칭한 스미싱 문자",
    )

    prompt = build_system_prompt(context, ["국민건강보험 언급", "즉시 확인 유도"])

    assert "90/100" in prompt
    assert "HIGH" in prompt
    assert "RiskGrade" not in prompt
    assert "기관 사칭형" in prompt
    assert "국민건강보험을 사칭한 스미싱 문자" in prompt
    assert "국민건강보험 언급, 즉시 확인 유도" in prompt


def test_build_system_prompt_handles_missing_phishing_type_and_indicators():
    context = AnalysisContext(
        riskScore=10,
        riskGrade="LOW",
        summary="일상적인 대화",
    )

    prompt = build_system_prompt(context, [])

    assert "미분류" in prompt
    assert "탐지 근거: 없음" in prompt
