from dataclasses import dataclass
from enum import Enum

from app.analysis.schemas import SmishingAnalysisResponse


class AnalysisExecutionStatus(str, Enum):
    """분석 실행 상태."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnalysisExecution:
    """분석 결과와 실패 트랙 정보를 담는 불변 데이터."""

    status: AnalysisExecutionStatus
    result: SmishingAnalysisResponse
    failed_tracks: tuple[str, ...]


def classify_execution(
    result: SmishingAnalysisResponse,
) -> AnalysisExecution:
    """AI 분석 응답을 실행 상태와 실패 트랙 목록으로 분류한다."""
    if result.status == "ERROR":
        return AnalysisExecution(
            status=AnalysisExecutionStatus.FAILED,
            result=result,
            failed_tracks=("PIPELINE",),
        )

    failed_tracks: list[str] = []

    text_analysis = result.text_analysis or {}
    text_result = text_analysis.get("result") or {}
    stage1_result = text_analysis.get("stage1_naive_bayes") or {}

    if (
        text_result.get("grade") == "UNKNOWN"
        or text_result.get("error_message")
    ):
        failed_tracks.append("TEXT")
    elif stage1_result.get("error_message"):
        failed_tracks.append("TEXT:NAIVE_BAYES")

    url_analysis = result.url_analysis or {}
    failed_providers = url_analysis.get("failed_providers") or []
    url_available = url_analysis.get("available", True)

    if result.url_analysis is not None and not url_available:
        failed_tracks.append("URL")
    else:
        failed_tracks.extend(
            f"URL:{provider}"
            for provider in failed_providers
        )

    rule_analysis = result.rule_analysis or {}
    if rule_analysis.get("error_message"):
        failed_tracks.append("RULES")

    status = (
        AnalysisExecutionStatus.PARTIAL
        if failed_tracks
        else AnalysisExecutionStatus.COMPLETED
    )

    return AnalysisExecution(
        status=status,
        result=result,
        failed_tracks=tuple(failed_tracks),
    )
