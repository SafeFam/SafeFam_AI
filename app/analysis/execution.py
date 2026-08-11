"""분석 파이프라인 결과의 실행 상태를 분류"""
from dataclasses import dataclass
from enum import Enum

from app.analysis.schemas import (
    SmishingAnalysisResponse,
)


class AnalysisExecutionStatus(str, Enum):
    """분석 파이프라인의 종합 실행 상태"""

    # 모든 분석 트랙이 정상적으로 실행
    COMPLETED = "COMPLETED"

    # 최종 결과는 생성됐지만 일부 엔진 또는 공급자가 실패
    PARTIAL = "PARTIAL"

    # 파이프라인 전체가 실패하여 정상 결과를 생성하지 못함
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnalysisExecution:
    """분석 결과와 실패 트랙 정보를 담는 불변 데이터"""

    status: AnalysisExecutionStatus

    # 실제 API 분석 결과
    result: SmishingAnalysisResponse

    # 실패한 분석 엔진 또는 외부 공급자 목록
    failed_tracks: tuple[str, ...]


def _classify_text_failures(
    text_analysis: dict,
) -> list[str]:
    """하이브리드 텍스트 분석기의 실패 엔진을 분류"""

    # HybridTextAnalyzer가 최종 선택한 텍스트 결과
    text_result = (
        text_analysis.get("result") or {}
    )

    # Stacking 자체 모델의 원본 결과
    self_model = (
        text_analysis.get("self_model") or {}
    )

    # 최종 텍스트 판정의 상태
    text_grade = text_result.get("grade")
    text_score = text_result.get("risk_score")
    text_error = text_result.get("error_message")

    # Stacking 자체 모델의 사용 가능 여부
    self_model_available = (
        self_model.get("risk_score") is not None
    )

    # Gemini가 실제로 호출됐는지 확인
    gemini_called = bool(
        text_analysis.get(
            "gemini_called",
            False,
        )
    )

    # Gemini 호출 결과를 최종 판정에 사용할 수 있었는지 확인
    gemini_available = bool(
        text_analysis.get(
            "gemini_available",
            False,
        )
    )

    # 최종 텍스트 결과 자체를 사용할 수 없는 경우
    text_failed = (
        not text_result
        or text_grade == "UNKNOWN"
        or text_score is None
        or bool(text_error)
    )

    if text_failed:
        # 두 엔진이 모두 실패한 경우 개별 엔진을 두 번 기록하기보다 텍스트 트랙 전체 실패인 TEXT 하나로 표현
        return ["TEXT"]

    failed_tracks: list[str] = []

    # Stacking은 실패했지만 Gemini가 성공하여 최종 텍스트 결과는 생성된 경우
    if not self_model_available:
        failed_tracks.append(
            "TEXT:STACKING"
        )

    # Gemini를 호출했지만 실패하고 Stacking 결과로 fallback한 경우
    if gemini_called and not gemini_available:
        failed_tracks.append(
            "TEXT:GEMINI"
        )

    return failed_tracks


def _classify_url_failures(
    url_analysis: dict | None,
) -> list[str]:
    """URL 분석 트랙과 공급자 실패를 분류"""

    if url_analysis is None:
        return []

    failed_tracks: list[str] = []

    failed_providers = (
        url_analysis.get("failed_providers") or []
    )

    url_available = bool(
        url_analysis.get(
            "available",
            True,
        )
    )

    if not url_available:
        # 모든 URL 공급자를 사용할 수 없는 경우 개별 공급자 대신 URL 트랙 전체 실패로 표현
        failed_tracks.append("URL")
    else:
        # 전체 URL 트랙은 사용 가능하지만 일부 공급자만 실패한 경우
        failed_tracks.extend(
            f"URL:{provider}"
            for provider in failed_providers
        )

    return failed_tracks


def _classify_rule_failures(
    rule_analysis: dict | None,
) -> list[str]:
    """로컬 규칙 분석기의 실패 여부를 분류"""

    if rule_analysis is None:
        return []

    if rule_analysis.get("error_message"):
        return ["RULES"]

    return []


def classify_execution(
    result: SmishingAnalysisResponse,
) -> AnalysisExecution:
    """분석 결과를 COMPLETED, PARTIAL, FAILED로 분류"""

    if result.status == "ERROR":
        return AnalysisExecution(
            status=AnalysisExecutionStatus.FAILED,
            result=result,
            failed_tracks=("PIPELINE",),
        )

    failed_tracks: list[str] = []

    # 텍스트 분석 실패 분류
    failed_tracks.extend(
        _classify_text_failures(
            result.text_analysis or {}
        )
    )

    # URL 분석 실패 분류
    failed_tracks.extend(
        _classify_url_failures(
            result.url_analysis
        )
    )

    # 규칙 분석 실패 분류
    failed_tracks.extend(
        _classify_rule_failures(
            result.rule_analysis
        )
    )

    # 실패한 트랙이 하나라도 있으면 PARTIAL 최종 분석 결과 자체는 생성됐기 때문에 FAILED로 처리하지 않음
    status = (
        AnalysisExecutionStatus.PARTIAL
        if failed_tracks
        else AnalysisExecutionStatus.COMPLETED
    )

    return AnalysisExecution(
        status=status,
        result=result,
        failed_tracks=tuple(
            failed_tracks
        ),
    )