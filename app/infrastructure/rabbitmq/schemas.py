from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# 분석 요청이 생성된 경로
class AnalysisSource(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"

# AI 분석에 필요한 실제 문자 데이터
class AnalysisRequestedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender: str | None = Field(default=None, max_length=100)
    content: str
    receivedAt: AwareDatetime
    source: AnalysisSource

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

# Spring이 발행하는 ANALYSIS_REQUESTED v1 이벤트
class AnalysisRequestedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["1.0"]
    eventId: UUID
    analysisId: int = Field(gt=0)
    clientMessageId: str | None = Field(
        default=None,
        max_length=100,
    )
    traceId: UUID
    occurredAt: AwareDatetime
    payload: AnalysisRequestedPayload