from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.analysis.schemas import RiskGrade


class AnalysisSource(str, Enum):
    """분석 요청이 생성된 경로를 정의"""

    AUTO = "AUTO"
    MANUAL = "MANUAL"


class AnalysisRequestedPayload(BaseModel):
    """AI 분석에 필요한 실제 문자 본문 데이터 스키마"""

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


class AnalysisRequestedEvent(BaseModel):
    """Spring 메시징 시스템이 발행하는 ANALYSIS_REQUESTED v1 이벤트 스키마"""

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


class AnalysisEventType(str, Enum):
    """분석 결과 이벤트의 종합 처리 상태를 정의"""

    COMPLETED = "ANALYSIS_COMPLETED"
    PARTIAL = "ANALYSIS_PARTIAL"
    FAILED = "ANALYSIS_FAILED"


class TextAnalysisMethod(str, Enum):
    """텍스트 분석에 사용된 AI 및 알고리즘 방식을 정의"""

    STACKING = "STACKING"
    LLM = "LLM"
    STACKING_LLM = "STACKING_LLM"
    STACKING_FALLBACK = "STACKING_FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"

    NAIVE_BAYES = "NAIVE_BAYES"
    NAIVE_BAYES_LLM = "NAIVE_BAYES_LLM"

    # deprecated
    GEMINI = "GEMINI"
    STACKING_GEMINI = "STACKING_GEMINI"


class RawScores(BaseModel):
    """각 분석 트랙별(텍스트, URL, 룰) 원시 점수 데이터 스키마"""

    model_config = ConfigDict(extra="forbid")

    text: int | None = Field(default=None, ge=0, le=100)
    url: int | None = Field(default=None, ge=0, le=100)
    rules: int | None = Field(default=None, ge=0, le=100)


class WeightedContributions(BaseModel):
    """최종 위험도 점수에 반영된 트랙별 가중치 기여 점수 스키마"""

    model_config = ConfigDict(extra="forbid")

    text: int = Field(ge=0, le=100)
    url: int = Field(ge=0, le=100)
    rules: int = Field(ge=0, le=100)


class TextAnalysisDetail(BaseModel):
    """텍스트 분석 트랙의 세부 진단 결과 스키마"""

    model_config = ConfigDict(extra="forbid")

    method: TextAnalysisMethod
    score: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    grade: str | None = None
    reason: str | None = None
    evidence: list[str] = Field(
        default_factory=list
    )
    failedEngines: list[str] = Field(
        default_factory=list
    )

    selfModelScore: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    selfModelConfidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    llmCalled: bool = False
    llmProvider: str | None = None
    llmModel: str | None = None
    llmUsage: dict[str, int | None] = Field(default_factory=dict)
    llmLatencyMs: float | None = Field(default=None, ge=0.0)
    llmFromCache: bool = False

    # Deprecated: temporary SafeFam_BE compatibility.
    geminiCalled: bool = False
    decisionSource: str | None = None
    routingReason: str | None = None
    fallbackApplied: bool = False


class UrlAnalysisDetail(BaseModel):
    """URL 분석 트랙의 세부 진단 결과 스키마"""

    model_config = ConfigDict(extra="forbid")

    hasUrl: bool
    originalUrl: str | None = None
    tracedUrl: str | None = None
    malicious: bool | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    engineSource: str | None = None
    errorCode: str | None = None


class RuleAnalysisDetail(BaseModel):
    """기반 룰 기반 탐지 트랙의 세부 진단 결과 스키마"""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    matchedRules: list[str] = Field(default_factory=list)
    maliciousDomainPattern: bool = False


class AnalysisResultPayload(BaseModel):
    """분석 결과 이벤트에 포함되는 통합 분석 페이로드 스키마"""

    model_config = ConfigDict(extra="forbid")

    finalScore: int | None = Field(default=None, ge=0, le=100)
    riskGrade: RiskGrade | None = None
    phishingType: str | None = None

    rawScores: RawScores
    weightedContributions: WeightedContributions | None = None

    textAnalysis: TextAnalysisDetail | None = None
    urlAnalysis: UrlAnalysisDetail | None = None
    ruleAnalysis: RuleAnalysisDetail | None = None

    failedTracks: list[str] = Field(default_factory=list)
    failureCode: str | None = None

    @model_validator(mode="after")
    def validate_failure_payload(
        self,
    ) -> "AnalysisResultPayload":
        if self.failureCode is not None and (
            self.finalScore is not None
            or self.riskGrade is not None
            or self.weightedContributions is not None
        ):
            raise ValueError("failure payload must not contain successful score fields")
        return self


class AnalysisResultEvent(BaseModel):
    """FastAPI가 처리 후 Spring으로 발행하는 분석 결과 이벤트 스키마"""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["1.0"]
    eventId: UUID
    causationId: UUID
    analysisId: int = Field(gt=0)
    clientMessageId: str | None = None
    traceId: UUID
    occurredAt: AwareDatetime
    eventType: AnalysisEventType
    payload: AnalysisResultPayload

    @model_validator(mode="after")
    def validate_event_result(self) -> "AnalysisResultEvent":
        """이벤트 타입(COMPLETED, PARTIAL, FAILED)에 따른 필드 유효성을 검증"""
        if self.eventType == AnalysisEventType.COMPLETED:
            if self.payload.finalScore is None:
                raise ValueError("completed event requires finalScore")
            if self.payload.riskGrade is None:
                raise ValueError("completed event requires riskGrade")
            if self.payload.failureCode is not None or self.payload.failedTracks:
                raise ValueError("completed event must not contain failure indicators")

        if self.eventType == AnalysisEventType.PARTIAL:
            if not self.payload.failedTracks:
                raise ValueError("partial event requires failedTracks")
            if self.payload.finalScore is None:
                raise ValueError("partial event requires finalScore")
            if self.payload.riskGrade is None:
                raise ValueError("partial event requires riskGrade")
            if self.payload.failureCode is not None:
                raise ValueError("partial event must not contain failureCode")

        if self.eventType == AnalysisEventType.FAILED:
            if not self.payload.failureCode:
                raise ValueError("failed event requires failureCode")
            if (
                self.payload.finalScore is not None
                or self.payload.riskGrade is not None
                or self.payload.weightedContributions is not None
            ):
                raise ValueError(
                    "failed event must not contain successful score fields"
                )

        return self


class DeadLetterEvent(BaseModel):
    """원문과 개인정보를 제외한 실패 메시지를 격리하는 이벤트 스키마"""

    model_config = ConfigDict(extra="forbid")

    schemaVersion: Literal["1.0"]
    eventId: UUID
    originalMessageId: str | None = Field(
        default=None,
        max_length=255,
    )
    analysisId: int | None = Field(
        default=None,
        gt=0,
    )
    traceId: UUID | None = None
    failureCode: str = Field(
        min_length=1,
        max_length=100,
    )
    failedAt: AwareDatetime
