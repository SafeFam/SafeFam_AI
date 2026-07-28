from datetime import datetime, timezone
from uuid import uuid4

from app.analysis.execution import (
    AnalysisExecution,
    AnalysisExecutionStatus,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisRequestedEvent,
    AnalysisResultEvent,
    AnalysisResultPayload,
)


class AnalysisResultEventFactory:
    """내부 분석 실행 결과를 메시징 규격 이벤트로 변환"""
    def create(
        self,
        request: AnalysisRequestedEvent,
        execution: AnalysisExecution,
    ) -> AnalysisResultEvent:
        """요청 이벤트 데이터와 실행 결과를 결합하여 전달할 결과 이벤트를 생성"""

        # 내부 분석 상태(COMPLETED, PARTIAL, FAILED)를 외부 메시징 이벤트 타입으로 맵핑
        event_type = {
            AnalysisExecutionStatus.COMPLETED:
                AnalysisEventType.COMPLETED,
            AnalysisExecutionStatus.PARTIAL:
                AnalysisEventType.PARTIAL,
            AnalysisExecutionStatus.FAILED:
                AnalysisEventType.FAILED,
        }[execution.status]

        result = execution.result

        # 원본 요청의 식별자를 포함하여 결과 이벤트 구성
        return AnalysisResultEvent(
            schemaVersion="1.0",
            eventId=uuid4(),
            causationId=request.eventId,
            analysisId=request.analysisId,
            clientMessageId=request.clientMessageId,
            traceId=request.traceId,
            occurredAt=datetime.now(timezone.utc),
            eventType=event_type,
            payload=build_payload(
                result=result,
                failed_tracks=execution.failed_tracks,
            ),
        )