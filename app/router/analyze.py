import logging
from fastapi import APIRouter, Depends, status, HTTPException
from app.dto.schemas import URLScanRequest, SmishingAnalysisResponse
from app.service.scan_service import ScanService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analyze"])

def get_scan_service() -> ScanService:
    return ScanService()

@router.post(
    "", 
    response_model=SmishingAnalysisResponse, 
    status_code=status.HTTP_200_OK,
    summary="[메인 통합 엔진] 문자 본문 기반 3중 스미싱 통합 분석"
)
async def analyze_smishing(
    payload: URLScanRequest,
    scan_service: ScanService = Depends(get_scan_service)
) -> SmishingAnalysisResponse:
    
    logger.info(f"[Router] 통합 스미싱 분석 마스터 파이프라인 진입: {payload.text[:15]}...")

    try:
        return await scan_service.analyze_pipeline(payload.text)
    except Exception as e:
        logger.error(f"[Router] 스캔 처리 중 장애 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"서버 내부 스캔 파이프라인 연산 중 오류: {str(e)}"
        )