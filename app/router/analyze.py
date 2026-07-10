from fastapi import APIRouter
from app.dto.response import ApiResponse

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("", response_model=ApiResponse[dict])
async def analyze_smishing(payload: dict):
    """
    스프링으로부터 메시지 및 URL 정보를 받아 분석을 수행하는 엔드포인트 초안
    """

    dummy_result = {
        "smishing_detected": False,
        "url_analysis": {
            "is_shortened": True,
            "origin_url": "https://safe-destination.com",
            "malicious_count": 0,
            "badge_color": "GREEN"
        }
    }
    return ApiResponse.success(data=dummy_result, message="분석이 완료되었습니다.")