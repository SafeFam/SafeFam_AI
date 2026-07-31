from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.analysis.schemas import RiskGrade


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class AnalysisContext(BaseModel):
    """Spring Boot가 /analyze 결과를 바탕으로 구성해 전달하는 분석 컨텍스트"""
    model_config = ConfigDict(extra="forbid")

    riskScore: int = Field(..., ge=0, le=100, description="최종 위험 점수 (0~100)")
    riskGrade: RiskGrade = Field(..., description="위험 등급 (HIGH/MEDIUM/LOW)")
    phishingType: str | None = Field(default=None, description="피싱 유형 (예: 기관 사칭형, 대출 사기형)")
    summary: str = Field(..., description="분석 결과 요약 설명")

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
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
    indicators: list[str] = Field(default_factory=list, description="탐지 근거 목록")
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
