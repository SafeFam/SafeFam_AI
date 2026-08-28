from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.schemas import RiskGrade
from app.core.config import settings


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Indicator(BaseModel):
    """탐지 근거 하나. type/description 값 목록이 아직 확정되지 않아 자유 문자열로 수용"""

    # hide_input_in_errors: 검증 실패 시 원문이 에러에 담기지 않도록 한다.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: str
    description: str

    @field_validator("type", "description")
    @classmethod
    def within_context_text_limit(cls, value: str) -> str:
<<<<<<< HEAD
        # LLM 프롬프트로 주입되므로 무제한 유입을 막는다(issue #120).
=======
        """LLM 프롬프트로 주입되는 근거 문자열의 무제한 유입을 막는다(issue #120)."""
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        if len(value) > settings.MAX_CHAT_CONTEXT_TEXT_LENGTH:
            raise ValueError("indicator field exceeds max length")
        return value


class AnalysisContext(BaseModel):
    """Spring Boot가 /analyze 결과를 바탕으로 구성해 전달하는 분석 컨텍스트"""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    riskScore: int = Field(..., ge=0, le=100, description="최종 위험 점수 (0~100)")
    riskLevel: RiskGrade = Field(..., description="위험 등급 (HIGH/MEDIUM/LOW)")
    category: str = Field(..., description="피싱 유형 분류 (예: FINANCIAL_INSTITUTION)")
    explanation: str = Field(..., description="분석 결과 설명")
    indicators: list[Indicator] = Field(
        default_factory=list, description="탐지 근거 목록"
    )

    @field_validator("category")
    @classmethod
    def category_within_limit(cls, value: str) -> str:
<<<<<<< HEAD
        # LLM 프롬프트로 주입되므로 무제한 유입을 막는다(issue #120).
=======
        """LLM 프롬프트로 주입되는 category의 무제한 유입을 막는다(issue #120)."""
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        if len(value) > settings.MAX_CHAT_CONTEXT_TEXT_LENGTH:
            raise ValueError("category exceeds max length")
        return value

    @field_validator("explanation")
    @classmethod
    def explanation_must_not_be_blank(cls, value: str) -> str:
        """공백이거나 상한을 초과하는 분석 요약을 거부한다(issue #120)."""
        if not value.strip():
            raise ValueError("explanation must not be blank")
        if len(value) > settings.MAX_CHAT_CONTEXT_TEXT_LENGTH:
            raise ValueError("explanation exceeds max length")
        return value

    @field_validator("indicators")
    @classmethod
    def indicators_within_limit(cls, value: list[Indicator]) -> list[Indicator]:
<<<<<<< HEAD
=======
        """탐지 근거 개수 상한을 강제해 프롬프트 팽창을 막는다(issue #120)."""
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        if len(value) > settings.MAX_CHAT_INDICATORS:
            raise ValueError("indicators exceeds max count")
        return value


class ChatMessage(BaseModel):
    # hide_input_in_errors: 검증 실패 시 챗 원문(PII)이 에러에 담기지 않도록 한다.
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """공백이거나 상한을 초과하는 챗 메시지를 거부한다(issue #120)."""
        if not value.strip():
            raise ValueError("content must not be blank")
        # BE ChatMessage @Size(max=2000)와 정합.
        if len(value) > settings.MAX_CHAT_CONTENT_LENGTH:
            raise ValueError("content exceeds max length")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    analysisContext: AnalysisContext | None = None
    messages: list[ChatMessage] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def messages_within_limit(cls, value: list[ChatMessage]) -> list[ChatMessage]:
<<<<<<< HEAD
=======
        """대화 메시지 개수 상한을 강제해 LLM 비용 폭증을 막는다(issue #120)."""
>>>>>>> f57ae1c1b7e725f9df63b20d31f6be5505de2a1c
        if len(value) > settings.MAX_CHAT_MESSAGES:
            raise ValueError("messages exceeds max count")
        return value


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
