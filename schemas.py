from pydantic import BaseModel, Field
from typing import Optional

# Spring Boot Gateway에서 Python FastAPI로 검사를 요청할 때의 바디 규격
class URLScanRequest(BaseMoel):

    text: str = Field(..., description="검사할 문자 메시지 본문 텍스트")

# Python FastAPI가 Spring Boot로 최종 전달할 하이브리드 검사 결과 규칙

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
