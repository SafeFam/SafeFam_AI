"""스미싱 분석 파이프라인 조립 서비스"""
import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.analysis.evidence import build_evidence
from app.analysis.hybrid_policy import (
    ConditionalLlmPolicy,
    HybridThresholds,
)
from app.analysis.risk_policy import (
    RISK_MEDIUM_THRESHOLD,
)
from app.analysis.rules.analyzer import (
    analyze_text_with_rules,
)
from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
)
from app.analysis.scoring import RiskScoringEngine
from app.analysis.text.llm_analyzer import (
    analyze_text_with_llm,
)
from app.analysis.text.hybrid_analyzer import (
    HybridTextAnalyzer,
)
from app.analysis.text.stacking_analyzer import (
    analyze_text_with_stacking,
)
from app.analysis.url.analyzer import (
    HybridUrlAnalyzer,
)
from app.analysis.url.tracker import (
    extract_urls,
    trace_url,
)
from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass
class _TrackOutcome:
    """텍스트/URL 트랙 실행 결과를 파이프라인 후속 단계로 전달하는 내부 상태.

    응답에 직렬화되지 않으며 오케스트레이션 단계 간 값 전달 용도로만 사용한다.
    """

    text_analysis: dict
    has_url: bool
    original_url: str | None
    traced_url: str | None
    hybrid_url_result: dict


@dataclass
class _TrackAvailability:
    """각 트랙의 사용 가능 여부와 스코어링 입력 점수를 담는 내부 상태."""

    text_available: bool
    scoring_text_score: int
    url_available: bool
    rules_available: bool


class SmishingAnalysisService:
    """텍스트, URL, 규칙 분석 결과를 최종 응답으로 조립"""

    def __init__(
        self,
        url_analyzer: HybridUrlAnalyzer | None = None,
        text_analyzer: HybridTextAnalyzer | None = None,
        rule_analyzer: (
            Callable[[str, str | None], dict] | None
        ) = None,
    ) -> None:
        """분석 서비스가 사용할 각 분석기를 구성"""

        self.url_analyzer = (
            url_analyzer or HybridUrlAnalyzer()
        )

        self.text_analyzer = (
            text_analyzer
            or HybridTextAnalyzer(

                # Stacking 확률을 기준으로
                # LLM 호출 여부를 결정하는 정책
                policy=ConditionalLlmPolicy(
                    HybridThresholds(
                        normal_max=(
                            settings
                            .STACKING_NORMAL_PROBABILITY_MAX
                        ),
                        phishing_min=(
                            settings
                            .STACKING_PHISHING_PROBABILITY_MIN
                        ),
                    )
                ),

                # 첫 번째 분석 엔진: 자체 stacking 모델
                stacking_analyzer=(
                    analyze_text_with_stacking
                ),

                # 두 번째 분석 엔진: 불확실한 경우에만 호출되는 LLM
                llm_analyzer=(
                    analyze_text_with_llm
                ),
            )
        )
        self.rule_analyzer = (
            rule_analyzer or analyze_text_with_rules
        )

    def _preview_rule_score(self, text: str) -> int:
        """LLM 강제 검증 여부를 판단할 규칙 점수를 계산."""

        try:
            result = self.rule_analyzer(text, None)
            score = result.get("rule_score", 0)

            if not isinstance(score, (int, float)):
                return RISK_MEDIUM_THRESHOLD

            return int(score)
        except Exception as exception:
            # 원문이나 예외 메시지는 로그에 남기지 않는다.
            logger.error(
                "[Analysis Service] Rule preview failed. "
                "error_type=%s",
                type(exception).__name__,
            )
            # 규칙 분석 실패를 안전 신호로 간주하지 않는다.
            return RISK_MEDIUM_THRESHOLD

    async def analyze_pipeline(
        self,
        text: str,
    ) -> SmishingAnalysisResponse:

        """텍스트, URL, 규칙 분석을 실행하고 최종 점수를 계산"""

        try:
            tracks = await self._run_tracks(text)

            rule_result = self._run_rules(text, tracks.traced_url)

            availability = self._assess_track_availability(
                text_analysis=tracks.text_analysis,
                has_url=tracks.has_url,
                hybrid_url_result=tracks.hybrid_url_result,
                rule_result=rule_result,
            )

            is_confirmed_malicious = self._apply_confirmed_malicious_boost(
                hybrid_url_result=tracks.hybrid_url_result,
                rule_result=rule_result,
            )

            # 텍스트, URL, 규칙이 전부 실패한 경우
            # 정상 또는 LOW 응답을 반환하면 안됨
            no_reliable_signal = (
                not availability.text_available
                and not availability.url_available
                and not availability.rules_available
            )

            if no_reliable_signal:
                raise ValueError(
                    "No reliable analysis signal is available"
                )

            (
                final_score,
                risk_grade,
                breakdown,
            ) = RiskScoringEngine.calculate_score(
                # HybridTextAnalyzer가 선택한 최종 텍스트 점수
                llm_score=availability.scoring_text_score,

                is_url_malicious=(
                    tracks.hybrid_url_result.get(
                        "is_malicious",
                        False,
                    )
                ),

                url_risk_score=(
                    tracks.hybrid_url_result.get(
                        "url_risk_score",
                        0.0,
                    )
                ),

                rule_score=rule_result.get(
                    "rule_score",
                    0,
                ),

                has_url=tracks.has_url,

                # HybridTextAnalyzer가 이미 하나의 최종 점수를
                # 선택했으므로 자체 모델 점수를 다시 혼합하지 않는다.
                naive_bayes_score=None,

                # 인자명은 기존 호환성을 유지하지만 실제 의미는
                # 선택된 텍스트 점수의 사용 가능 여부다.
                llm_available=availability.text_available,

                is_confirmed_malicious=(
                    is_confirmed_malicious
                ),

                text_available=availability.text_available,
                url_available=availability.url_available,
                rules_available=availability.rules_available,
            )

            return self._assemble_success_response(
                has_url=tracks.has_url,
                original_url=tracks.original_url,
                traced_url=tracks.traced_url,
                hybrid_url_result=tracks.hybrid_url_result,
                url_available=availability.url_available,
                final_score=final_score,
                risk_grade=risk_grade,
                breakdown=breakdown,
                rule_result=rule_result,
                text_analysis=tracks.text_analysis,
            )

        except Exception as exception:
            # 예외 타입만 로그에 기록
            logger.error(
                "[Analysis Service] 파이프라인 실패. "
                "error_type=%s",
                type(exception).__name__,
            )

            return self._build_failure_response()

    async def _run_tracks(
        self,
        text: str,
    ) -> _TrackOutcome:
        """텍스트/URL 트랙을 병렬 실행하고 결과를 회수한다."""

        # 메시지에서 URL을 먼저 추출
        urls = extract_urls(text)
        has_url = len(urls) > 0

        rule_score_preview = self._preview_rule_score(
            text
        )
        force_llm = (
            rule_score_preview
            >= RISK_MEDIUM_THRESHOLD
        )

        # 텍스트 분석은 항상 실행
        text_task = asyncio.create_task(
            self.text_analyzer.analyze(
                text,
                force_llm=force_llm,
            )
        )

        url_task = None
        original_url = None

        # URL이 있는 경우에만 URL 추적 및 보안 분석 실행
        if has_url:
            original_url = urls[0]

            async def url_track():
                """단축 URL을 추적한 뒤 보안 엔진으로 검사"""

                traced = await trace_url(
                    original_url
                )

                analysis = (
                    await self.url_analyzer.scan_url(
                        traced
                    )
                )

                return traced, analysis

            url_task = asyncio.create_task(
                url_track()
            )

        # 텍스트와 URL 분석을 가능한 한 병렬로 실행
        if url_task is not None:
            try:
                (
                    text_analysis,
                    (
                        traced_url,
                        hybrid_url_result,
                    ),
                ) = await asyncio.gather(
                    text_task,
                    url_task,
                )
            except BaseException:
                # 한 트랙이 실패하면 아직 실행 중인 형제 task를
                # 취소하고 두 결과를 모두 회수해 orphan task와
                # "Task exception was never retrieved"를 방지한다.
                for task in (text_task, url_task):
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    text_task,
                    url_task,
                    return_exceptions=True,
                )
                raise
        else:
            text_analysis = await text_task

            traced_url = None

            # URL이 없는 것은 URL 분석 실패가 아님
            # 단순히 URL 트랙이 적용되지 않은 상태
            hybrid_url_result = {
                "is_malicious": False,
                "url_risk_score": 0.0,
                "source": (
                    "Pre-Processing-Filter"
                ),
                "available": False,
                "failed_providers": [],
                "pending_providers": [],
                "provider_error_codes": {},
                "error_message": None,
                "is_gsb_confirmed": False,
                "is_vt_confirmed": False,
            }

        return _TrackOutcome(
            text_analysis=text_analysis,
            has_url=has_url,
            original_url=original_url,
            traced_url=traced_url,
            hybrid_url_result=hybrid_url_result,
        )

    def _run_rules(
        self,
        text: str,
        traced_url: str | None,
    ) -> dict:
        """로컬 규칙 분석을 실행하고 실패 시 안전한 기본 결과를 반환한다."""

        try:
            return self.rule_analyzer(
                text,
                traced_url,
            )
        except Exception as exception:
            # 원문 메시지나 예외 메시지는 로그에 기록 X
            logger.error(
                "[Analysis Service] 규칙 분석 실패. "
                "error_type=%s",
                type(exception).__name__,
            )

            return {
                "rule_score": 0,
                "has_malicious_domain_pattern": (
                    False
                ),
                "matched_rules": [],
                "error_message": (
                    "RULE_ANALYSIS_FAILED"
                ),
            }

    def _assess_track_availability(
        self,
        *,
        text_analysis: dict,
        has_url: bool,
        hybrid_url_result: dict,
        rule_result: dict,
    ) -> _TrackAvailability:
        """각 트랙의 사용 가능 여부와 스코어링 입력 점수를 판정한다."""

        # HybridTextAnalyzer가 선택한 최종 텍스트 결과
        text_result = (
            text_analysis.get("result") or {}
        )

        # 하이브리드 분석기가 최종 선택한 텍스트 점수
        selected_text_score = (
            text_result.get("risk_score")
        )

        # 두 텍스트 엔진이 모두 실패한 경우 scoring engine이 fail-safe 점수를 적용할 수 있도록 0을 전달
        scoring_text_score = (
            int(selected_text_score)
            if selected_text_score is not None
            else 0
        )

        # HybridTextAnalyzer가 선택한 단일 결과가 있을 때만
        # 텍스트 트랙을 사용 가능한 상태로 본다.
        text_available = selected_text_score is not None

        rules_available = not bool(
            rule_result.get("error_message")
        )

        # URL이 존재하면서 URL 보안 공급자 중 하나 이상이
        # 정상 결과를 제공했을 때만 URL 트랙을 available로 봄
        url_available = (
            has_url
            and hybrid_url_result.get(
                "available",
                False,
            )
        )

        return _TrackAvailability(
            text_available=text_available,
            scoring_text_score=scoring_text_score,
            url_available=url_available,
            rules_available=rules_available,
        )

    def _apply_confirmed_malicious_boost(
        self,
        *,
        hybrid_url_result: dict,
        rule_result: dict,
    ) -> bool:
        """확정 악성 신호를 반영해 URL 위험도를 보정하고 확정 여부를 반환한다."""

        # 로컬 도메인 규칙에서 명확한 악성 패턴이 발견되면
        # URL 결과의 최소 위험도를 0.75로 올림
        if rule_result.get(
            "has_malicious_domain_pattern",
            False,
        ):
            hybrid_url_result[
                "is_malicious"
            ] = True

            hybrid_url_result[
                "url_risk_score"
            ] = max(
                hybrid_url_result.get(
                    "url_risk_score",
                    0.0,
                ),
                0.75,
            )

        # 신뢰도가 높은 확정 악성 신호를 확인
        # 다음 신호 중 하나라도 있으면 scoring engine에서 최종 등급을 최소 HIGH로 보정
        is_confirmed_malicious = (
            hybrid_url_result.get(
                "is_gsb_confirmed",
                False,
            )
            or hybrid_url_result.get(
                "is_vt_confirmed",
                False,
            )
            or rule_result.get(
                "has_malicious_domain_pattern",
                False,
            )
        )

        return is_confirmed_malicious

    def _assemble_success_response(
        self,
        *,
        has_url: bool,
        original_url: str | None,
        traced_url: str | None,
        hybrid_url_result: dict,
        url_available: bool,
        final_score: int,
        risk_grade: RiskGrade,
        breakdown: ContributionBreakdown,
        rule_result: dict,
        text_analysis: dict,
    ) -> SmishingAnalysisResponse:
        """분석 결과를 최종 응답 스키마로 조립한다."""

        # URL이 실제로 포함된 경우에만
        # URL 분석 상세 결과를 응답에 포함
        if has_url:
            real_url_analysis = {
                "has_url": True,

                "is_shortened": (
                    original_url != traced_url
                ),

                "origin_url": traced_url,
                "original_url": original_url,

                "is_url_malicious": (
                    hybrid_url_result.get(
                        "is_malicious",
                        False,
                    )
                ),

                "url_risk_score": (
                    hybrid_url_result.get(
                        "url_risk_score",
                        0.0,
                    )
                ),

                "engine_source": (
                    hybrid_url_result.get(
                        "source",
                        "Hybrid-Engine",
                    )
                ),

                "available": url_available,

                "failed_providers": (
                    hybrid_url_result.get(
                        "failed_providers",
                        [],
                    )
                ),

                "pending_providers": (
                    hybrid_url_result.get(
                        "pending_providers",
                        [],
                    )
                ),

                "provider_error_codes": (
                    hybrid_url_result.get(
                        "provider_error_codes",
                        {},
                    )
                ),

                "error_message": (
                    hybrid_url_result.get(
                        "error_message"
                    )
                ),
            }
        else:
            real_url_analysis = None

        # 원문 메시지를 응답 또는 로그에 추가 X
        return SmishingAnalysisResponse(
            status="SUCCESS",
            message=(
                "3중 가중치 결합 스미싱 "
                "통합 분석이 완료되었습니다."
            ),
            final_score=final_score,
            risk_grade=risk_grade,
            contribution_breakdown=breakdown,
            evidence=build_evidence(
                rule_analysis=rule_result,
                url_analysis=real_url_analysis,
                text_analysis=text_analysis,
            ),
            text_analysis=text_analysis,
            url_analysis=real_url_analysis,
            rule_analysis=rule_result,
        )

    def _build_failure_response(
        self,
    ) -> SmishingAnalysisResponse:
        """파이프라인 실패 시 fail-open을 막는 안전 응답을 반환한다."""

        # 전체 파이프라인 실패를 0점/LOW로 반환하면
        # 장애가 안전 판정으로 해석되는 fail-open이 발생
        return SmishingAnalysisResponse(
            status="ERROR",
            message=(
                "분석 파이프라인 처리 중 "
                "오류가 발생했습니다."
            ),
            final_score=(
                RiskScoringEngine
                .PIPELINE_FAILURE_FALLBACK_SCORE
            ),
            risk_grade=RiskGrade.MEDIUM,
            contribution_breakdown=(
                ContributionBreakdown(
                    llm=0,
                    hybrid_url=0,
                    rules=0,
                )
            ),
            text_analysis=None,
            url_analysis=None,
            rule_analysis=None,
        )
