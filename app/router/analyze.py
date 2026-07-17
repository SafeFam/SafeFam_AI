from fastapi import APIRouter
from app.dto.response import ApiResponse
from app.dto.request import AnalyzeRequest
from app.service.security.gemini_text_analyzer import analyze_text_with_gemini

# 문자 분석 전용 라우터 생성
router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("", response_model=ApiResponse[dict])
async def analyze_smishing(payload: AnalyzeRequest):
    """
    스프링으로부터 메시지를 받아 Gemini 기반 어조/근거 분석으로 문자 위험도를 산출하는 엔드포인트
    """

    text_analysis = await analyze_text_with_gemini(payload.message)

    # URL 분석 파이프라인은 별도 작업으로 추후 연결 예정 (더미 유지)
    dummy_url_analysis = {
        "is_shortened": True, # 단축 URL 여부
        "origin_url": "https://safe-destination.com", # 최종 목적지 URL
        "malicious_count": 0,   # 악성 판전 횟수
        "badge_color": "GREEN"
    }

    result = {
        "smishing_detected": text_analysis["result"]["grade"] != "SAFE",
        "text_analysis": text_analysis,
        "url_analysis": dummy_url_analysis
    }
    return ApiResponse.success(data=result, message="문자 분석이 완료되었습니다.")