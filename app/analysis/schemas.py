from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings


# Spring Boot Gateway에서 Python FastAPI로 검사를 요청할 때의 바디 규격
class SmishingAnalysisRequest(BaseModel):
    # hide_input_in_errors: 검증 실패 시 원문(PII)이 에러에 담기지 않도록 한다.
    model_config = ConfigDict(hide_input_in_errors=True)

    text: str = Field(..., description="검사할 문자 메시지 본문 텍스트")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        # 분석 content와 동일한 상한(BE @Size(max=5000)와 정합)을 공유한다.
        if len(value) > settings.MAX_ANALYSIS_CONTENT_LENGTH:
            raise ValueError("text exceeds max length")
        return value


# Python FastAPI가 Spring Boot로 최종 전달할 하이브리드 검사 결과 규칙
class UrlAnalysisResponse(BaseModel):
    has_url: bool = Field(..., description="본문 내 URL 포함 여부")
    original_url: str | None = Field(None, description="최초 추출된 URL")
    traced_url: str | None = Field(None, description="단축 URL 추적 결과 (최종 목적지)")
    is_url_malicious: bool = Field(..., description="최종 악성 URL 판정 여부")
    url_risk_score: float = Field(..., description="위험도 점수 (0.0 ~ 1.0)")
    engine_source: str = Field(
        ..., description="분석 엔진 소스 정보 (예: 'Hybrid-Engine')"
    )
    error_message: str | None = Field(None, description="에러 발생 시 메시지 기록용")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "has_url": True,
                "original_url": "https://bit.ly/suspect-link",
                "traced_url": "https://malicious-phishing-site.com/login",
                "is_url_malicious": True,
                "url_risk_score": 0.85,
                "engine_source": "Hybrid-Engine",
                "error_message": None,
            }
        }
    )


# 3중 가중치 시스템 스펙 정의
class RiskGrade(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EvidenceCategory(str, Enum):
    """증거 카드(issue #64) 분류. 사용자에게는 title/description만 노출되며,
    내부 모델/엔진 이름은 절대 이 값이나 설명 문구에 담기지 않는다."""

    INSTITUTION_IMPERSONATION = "INSTITUTION_IMPERSONATION"
    PERSONAL_INFO_REQUEST = "PERSONAL_INFO_REQUEST"
    DANGEROUS_URL = "DANGEROUS_URL"
    URGENCY_PRESSURE = "URGENCY_PRESSURE"
    AI_JUDGMENT = "AI_JUDGMENT"


class EvidenceItem(BaseModel):
    """탐지된 위험 신호 하나를 사용자 언어로 요약한 증거 카드 한 장"""

    category: EvidenceCategory
    title: str = Field(..., description="카드 제목 (예: '기관 사칭')")
    description: str = Field(..., description="사용자가 이해할 수 있는 근거 설명")


class ContributionBreakdown(BaseModel):
    # URL이 있으면 LLM 50% / 규칙 20%, URL이 없으면 하이브리드 URL 트랙(30%)이 LLM/규칙으로
    # 재배분되어 LLM 65% / 규칙 35%가 되므로 상한이 두 시나리오 중 더 큰 쪽 기준으로 설정됨
    llm: int = Field(ge=0, le=100)
    hybrid_url: int = Field(ge=0, le=100)
    rules: int = Field(ge=0, le=100)


class SmishingAnalysisResponse(BaseModel):
    status: str = Field(..., description="응답 상태 (SUCCESS / ERROR)")
    message: str = Field(..., description="응답 메시지 설명")

    # 핵심 합성 스코어 필드
    final_score: int = Field(
        ..., description="3중 가중치 합성 최종 위험 점수 (0~100)", ge=0, le=100
    )
    risk_grade: RiskGrade = Field(
        ..., description="최종 점수 기반 위험 등급 분류 (HIGH/MEDIUM/LOW)"
    )
    contribution_breakdown: ContributionBreakdown = Field(
        ..., description="3개 레이어별 점수 기여도 명세"
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="탐지된 위험 신호를 사용자 언어로 요약한 증거 카드 목록",
    )

    # 세부 분석 트랙 데이터
    text_analysis: dict | None = Field(
        None, description="LLM 실시간 문맥 분석 상세 결과"
    )
    url_analysis: dict | None = Field(
        None, description="하이브리드 URL 보안 엔진 상세 분석 결과"
    )
    rule_analysis: dict | None = Field(
        None,
        description="로컬 규칙 기반 엔진(금융기관 DB/키워드/계좌·카드번호) 상세 분석 결과",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "status": "SUCCESS",
                "message": "통합 스미싱 분석이 완료되었습니다.",
                "final_score": 95,
                "risk_grade": "HIGH",
                "contribution_breakdown": {"llm": 45, "hybrid_url": 30, "rules": 20},
                "text_analysis": {
                    "risk_score": 90,
                    "reason": "지인을 사칭한 금전 요구 문맥 감지",
                },
                "url_analysis": {
                    "has_url": True,
                    "is_url_malicious": True,
                    "url_risk_score": 0.95,
                },
            }
        },
    )
