import logging

from app.analysis.schemas import SmishingAnalysisResponse
from app.analysis.service import SmishingAnalysisService
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
)

logger = logging.getLogger(__name__)


class AnalysisRequestHandler:
    """분석 요청 이벤트를 AI 분석 파이프라인에 연결"""

    def __init__(
        self,
        analysis_service: SmishingAnalysisService,
    ) -> None:
        self.analysis_service = analysis_service

    async def handle(
        self,
        event: AnalysisRequestedEvent,
    ) -> SmishingAnalysisResponse:
        """분석 요청을 실행하고 파이프라인 응답을 그대로 반환"""
        logger.info(
            "Starting analysis request processing. "
            "event_id=%s analysis_id=%s trace_id=%s",
            event.eventId,
            event.analysisId,
            event.traceId,
        )

        result = await self.analysis_service.analyze_pipeline(
            event.payload.content
        )

        if result.status == "ERROR":
            logger.warning(
                "Analysis pipeline returned a failed result. "
                "event_id=%s analysis_id=%s trace_id=%s",
                event.eventId,
                event.analysisId,
                event.traceId,
            )

            return result

        logger.info(
            "Analysis request processing completed. "
            "event_id=%s analysis_id=%s trace_id=%s "
            "final_score=%s risk_grade=%s",
            event.eventId,
            event.analysisId,
            event.traceId,
            result.final_score,
            result.risk_grade,
        )

        return result
