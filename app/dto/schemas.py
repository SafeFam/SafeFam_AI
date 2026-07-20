from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

# Spring Boot Gateway에서 Python FastAPI로 검사를 요청할 때의 바디 규격
class URLScanRequest(BaseModel):
    text: str = Field(..., description="검사할 문자 메시지 본문 텍스트")

# Python FastAPI가 Spring Boot로 최종 전달할 하이브리드 검사 결과 규칙
class URLScanResponse(BaseModel):
    has_url: bool = Field(..., description="본문 내 URL 포함 여부")
    original_url: Optional[str] = Field(None, description="최초 추출된 URL")
    traced_url: Optional[str] = Field(None, description="단축 URL 추적 결과 (최종 목적지)")
    is_url_malicious: bool = Field(..., description="최종 악성 URL 판정 여부")
    url_risk_score: float = Field(..., description="위험도 점수 (0.0 ~ 1.0)")
    engine_source: str = Field(..., description="분석 엔진 소스 정보 (예: 'Hybrid-Engine')")
    error_message: Optional[str] = Field(None, description="에러 발생 시 메시지 기록용")

    class Config:
        json_schema_extra = {
            "example": {
                "has_url": True,
                "original_url": "https://bit.ly/suspect-link",
                "traced_url": "https://malicious-phishing-site.com/login",
                "is_url_malicious": True,
                "url_risk_score": 0.85,
                "engine_source": "Hybrid-Engine",
                "error_message": None
            }
        } 

# 3중 가중치 시스템 스펙 정의
class RiskGrade(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ContributionBreakdown(BaseModel):
    llm: int = Field(..., description="LLM 문맥 분석 기여 점수 (0~50)", ge=0, le=50)
    hybrid_url: int = Field(..., description="하이브리드 URL 보안 엔진 기여 점수 (0~30)", ge=0, le=30)
    rules: int = Field(..., description="로컬 가드 규칙 기반 기여 점수 (0~20)", ge=0, le=20)

class SmishingAnalysisResponse(BaseModel):
    status: str = Field(..., description="응답 상태 (SUCCESS / ERROR)")
    message: str = Field(..., description="응답 메시지 설명")

    # 핵심 합성 스코어 필드
    final_score: int = Field(..., description="3중 가중치 합성 최종 위험 점수 (0~100)", ge=0, le=100)
    risk_grade: RiskGrade = Field(..., description="최종 점수 기반 위험 등급 분류 (HIGH/MEDIUM/LOW)")
    contribution_breakdown: ContributionBreakdown = Field(..., description="3개 레이어별 점수 기여도 명세")

    # 세부 분석 트랙 데이터 
    text_analysis: Optional[dict] = Field(None, description="LLM 실시간 문맥 분석 상세 결과")
    url_analysis: Optional[dict] = Field(None, description="하이브리드 URL 보안 엔진 상세 분석 결과")

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "status": "SUCCESS",
                "message": "통합 스미싱 분석이 완료되었습니다.",
                "final_score": 95,
                "risk_grade": "HIGH",
                "contribution_breakdown": {
                    "llm": 45,
                    "hybrid_url": 30,
                    "rules": 20
                },
                "text_analysis": {
                    "risk_score": 90,
                    "reason": "지인을 사칭한 금전 요구 문맥 감지"
                },
                "url_analysis": {
                    "has_url": True,
                    "is_url_malicious": True,
                    "url_risk_score": 0.95
                }
            }
        }