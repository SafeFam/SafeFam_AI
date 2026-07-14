from fastapi import APIRouter
from app.dto.response import ApiResponse

# 음성 분석 전용 라우터 생성
router = APIRouter(prefix="/analyze-voice", tags=["Analyze Voice"])

@router.post("", response_model=ApiResponse[dict])
async def analyze_voice_call(payload: dict):
    """
    스프링으로부터 음성 데이터 또는 STT 텍스트 정보를 받아
    보이스피싱 및 악성 문맥 분석을 수행하는 엔드포인트 초안
    """

    # 더미 결과 구조
    dummy_result = {
        "voice_phishing_detected": False,
        "risk_score": 15,           # 보이스피싱 위험도 점수
        "detected_keywords": [],     # 탐지된 금융 사기 관련 키워드 목록
        "analysis_summary": "현재 통화 문맥상 금융 사기 및 피싱 징후가 발견되지 않은 안전한 상태입니다."
    }

    return ApiResponse.success(data=dummy_result, message="음성 분석이 완료되었습니다.")