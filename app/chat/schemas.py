from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.schemas import RiskGrade


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Indicator(BaseModel):
    """탐지 근거 하나. type/description 값 목록이 아직 확정되지 않아 자유 문자열로 수용"""

    model_config = ConfigDict(extra="forbid")

    type: str
    description: str


class AnalysisContext(BaseModel):
    """Spring Boot가 /analyze 결과를 바탕으로 구성해 전달하는 분석 컨텍스트"""

    model_config = ConfigDict(extra="forbid")

    riskScore: int = Field(..., ge=0, le=100, description="최종 위험 점수 (0~100)")
    riskLevel: RiskGrade = Field(..., description="위험 등급 (HIGH/MEDIUM/LOW)")
    category: str = Field(..., description="피싱 유형 분류 (예: FINANCIAL_INSTITUTION)")
    explanation: str = Field(..., description="분석 결과 설명")
    indicators: list[Indicator] = Field(
        default_factory=list, description="탐지 근거 목록"
    )

    @field_validator("explanation")
    @classmethod
    def explanation_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explanation must not be blank")
        return value


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysisContext: AnalysisContext
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
