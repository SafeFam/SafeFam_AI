import logging
from fastapi import APIRouter, Depends, status
from app.dto.request import AnalyzeRequest
from app.dto.schemas import SmishingAnalysisResponse
from app.service.scan_service import ScanService

logger = logging.getLogger(__name__)

# 문자 분석 전용 라우터 생성
router = APIRouter(prefix="/analyze", tags=["Analyze"])

# 서비스 인스턴스 생성 유틸
def get_scan_service() -> ScanService:
    return ScanService()

# 통합 스미싱 탐지 API
@router.post(
    "", 
    response_model=SmishingAnalysisResponse, 
    status_code=status.HTTP_200_OK,
    summary="[메인 통합 엔진] 문자 본문 기반 3중 스미싱 통합 분석",
    description="문자 본문 전체를 분석하여 LLM 문맥, 하이브리드 URL 검사, 로컬 규칙을 합성한 0~100점 점수를 반환합니다."
)
async def analyze_smishing(
    payload: AnalyzeRequest,
    scan_service: ScanService = Depends(get_scan_service)
    ) -> SmishingAnalysisResponse:
    
    # Spring Boot에서 전달된 문제 메시지를 접수하여 비동기 파이프라인(LLM + 하이브리드 URL + 로컬 룰)으로 정밀 스캔
    
    logger.info(f"[Router] 통합 스미싱 분석 마스터 파이프라인 진입: {payload.message[:15]}...")

    return await scan_service.analyze_pipeline(payload.message)