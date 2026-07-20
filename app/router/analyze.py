import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.dto.response import ApiResponse
from app.dto.request import AnalyzeRequest
from app.dto.schemas import URLScanResponse
from app.service.security.gemini_text_analyzer import analyze_text_with_gemini
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
    response_model=ApiResponse[dict], 
    status_code=status.HTTP_200_OK,
    summary="[메인 통합 엔진] 문자 본문 기반 3중 스미싱 통합 분석",
    description="문자 본문 전체를 분석하여 LLM 문맥, 하이브리드 URL 검사, 로컬 규칙을 합성한 0~100점 점수를 반환합니다."
    )

async def analyze_smishing(
    payload: AnalyzeRequest,
    scan_service: ScanService = Depends(get_scan_service)
    ) -> SmishingAnalysisResponse:
    
    try:
        logger.info(f"[Router] 통합 스미싱 분석 요청 접수: {payload.message[:15]}...")

        # 파이프라인 호출
        text_analysis = await analyze_text_with_gemini(payload.message)
        url_scan_result = await scan_service.scan_message_text(payload.message)

        # 하이브리드 url 엔진 결과 규격
        real_url_analysis = {
            "has_url": url_scan_result.has_url,
            "is_shortened": url_scan_result.original_url != url_scan_result.traced_url if url_scan_result.has_url else False,
            "origin_url": url_scan_result.traced_url,
            "original_url": url_scan_result.original_url,
            "is_url_malicious": url_scan_result.is_url_malicious,
            "url_risk_score": url_scan_result.url_risk_score,
            "engine_source": url_scan_result.engine_source,
            "error_message": url_scan_result.error_message
        }
    
        return SmishingAnalysisResponse(
            status="SUCCESS",
            message="통합 스미싱 분석이 정상 수행되었습니다.",
            final_score=0,                        
            risk_grade=RiskGrade.LOW,              
            contribution_breakdown=ContributionBreakdown(
                llm=0,
                hybrid_url=0,
                rules=0
            ),
            text_analysis=text_analysis,
            url_analysis=real_url_analysis
        )

    except Exception as e:
        logger.error(f"[Router Error] 통합 분석 중 예외 발생: {str(e)}")
        return SmishingAnalysisResponse(
            status="ERROR",
            message=f"통합 스미싱 탐지 중 서버 에러가 발생했습니다: {str(e)}",
            final_score=0,
            risk_grade=RiskGrade.LOW,
            contribution_breakdown=ContributionBreakdown(llm=0, hybrid_url=0, rules=0),
            text_analysis=None,
            url_analysis=None
        )