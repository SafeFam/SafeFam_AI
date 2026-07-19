from fastapi import APIRouter, Depends, HTTPException, status
from app.dto.response import ApiResponse
from app.dto.request import AnalyzeRequest
from app.dto.schemas import URLScanResponse
from app.service.security.gemini_text_analyzer import analyze_text_with_gemini
from app.service.scan_service import ScanService

# 문자 분석 전용 라우터 생성
router = APIRouter(prefix="/analyze", tags=["Analyze"])

# 서비스 인스턴스 생성 유틸
def get_scan_service() -> ScanService:
    return ScanService()

# 통합 스미싱 탐지 API
@router.post("", response_model=ApiResponse[dict], status_code=status.HTTP_200_OK)
async def analyze_smishing(
    payload: AnalyzeRequest,
    scan_service: ScanService = Depends(get_scan_service)
    ):
    
    try:
        text_analysis = await analyze_text_with_gemini(payload.message)

        url_scan_result: URLScanResponse = await scan_service.scan_message_text(payload.message)

        real_url_analysis = {
            "has_url": url_scan_result.has_url,
            "is_shortened": url_scan_result.original_url != url_scan_result.traced_url if url_scan_result.has_url else False,
            "origin_url": url_scan_result.traced_url,  # 최종 목적지 URL
            "original_url": url_scan_result.original_url, # 최초 추출 URL
            "is_url_malicious": url_scan_result.is_url_malicious,
            "url_risk_score": url_scan_result.url_risk_score,
            "engine_source": url_scan_result.engine_source,
            "error_message": url_scan_result.error_message
        }
    
        is_smishing_detected = (text_analysis.get("result", {}).get("grade") != "SAFE") or url_scan_result.is_url_malicious

        result = {
            "smishing_detected": is_smishing_detected,
            "text_analysis": text_analysis,
            "url_analysis": real_url_analysis
        }

        return ApiResponse.success(data=result, message="문자 분석이 완료되었습니다.")

    except Exception as e:
        return ApiResponse.error(message=f"통합 스미싱 탐지 중 서버 에러가 발생했습니다: {str(e)}")
