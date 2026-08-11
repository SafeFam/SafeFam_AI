import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.analysis.risk_policy import RISK_MEDIUM_THRESHOLD
from app.analysis.rules.analyzer import analyze_text_with_rules
from app.analysis.schemas import (
    ContributionBreakdown,
    RiskGrade,
    SmishingAnalysisResponse,
    UrlAnalysisResponse,
)
from app.analysis.scoring import RiskScoringEngine
from app.analysis.text.gemini_analyzer import analyze_text_with_gemini
from app.analysis.text.naive_bayes_analyzer import analyze_text_with_naive_bayes
from app.analysis.url.analyzer import HybridUrlAnalyzer
from app.analysis.url.tracker import extract_urls, trace_url

logger = logging.getLogger(__name__)


class SmishingAnalysisService:
    def __init__(
        self,
        url_analyzer: HybridUrlAnalyzer | None = None,
        naive_bayes_analyzer: Callable[[str], Awaitable[dict]] | None = None,
        gemini_analyzer: Callable[[str], Awaitable[dict]] | None = None,
        rule_analyzer: Callable[[str, str | None], dict] | None = None,
    ):
        self.url_analyzer = url_analyzer or HybridUrlAnalyzer()
        self.naive_bayes_analyzer = (
            naive_bayes_analyzer or analyze_text_with_naive_bayes
        )
        self.gemini_analyzer = gemini_analyzer or analyze_text_with_gemini
        self.rule_analyzer = rule_analyzer or analyze_text_with_rules

    # 1차 나이브 베이즈 선별 후, SAFE 기준 미만(의심)일 때만 Gemini 2차 검증을 호출하는 하이브리드 텍스트 트랙
    # 반환값: (text_analysis 응답용 dict, 나이브 베이즈 점수(미수행 시 None), Gemini 2차 검증 정상 수행 여부)
    async def _analyze_text_hybrid(self, text: str, rule_score: int) -> tuple[dict, int | None, bool]:
        nb_result = await self.naive_bayes_analyzer(text)
        nb_grade = nb_result.get("result", {}).get("grade")
        nb_score = nb_result.get("result", {}).get("risk_score", 0)
        nb_available = nb_result.get("is_available", False)

        # 모델이 정상 로드되어 SAFE로 판정한 경우에만 Gemini 호출 스킵.
        # 모델 로드 실패(UNKNOWN)는 안전하게 Gemini 2차 검증으로 폴백(fail-safe).
        # 단, 규칙 엔진이 이미 SUSPICIOUS 이상(계좌번호/기관명/긴급 키워드 등)을 감지했다면
        # NB 단독의 SAFE 판정만으로 스킵하지 않는다 — "정중한 공지문처럼 문체만 바꾼" 변형이
        # NB를 속여 SAFE로 오판시키고 Gemini 2차 검증 자체를 건너뛰게 만드는 것을 방지한다.
        if nb_available and nb_grade == "SAFE" and rule_score < RISK_MEDIUM_THRESHOLD:
            logger.info("[Hybrid] 나이브 베이즈 SAFE 판정 - Gemini 2차 검증 스킵")
            return nb_result, nb_score, False

        text_analysis = await self.gemini_analyzer(text)
        text_analysis["stage1_naive_bayes"] = nb_result.get("result")

        llm_available = text_analysis.get("result", {}).get("grade") != "UNKNOWN"
        return text_analysis, (nb_score if nb_available else None), llm_available

    # 텍스트 트랙과 URL 트랙을 병렬 조립하고 스코어링을 매핑
    async def analyze_pipeline(self, text: str) -> SmishingAnalysisResponse:

        try:
            urls = extract_urls(text)
            has_url = len(urls) > 0

            # NB 스킵 여부 판단에 쓸 규칙 점수를 URL 추적 이전에 미리 계산 (텍스트만으로 계산 가능한
            # 계좌/기관명/긴급 키워드 신호면 충분 — 도메인 룰은 traced_url 확정 후 최종 rule_result에서 반영)
            try:
                rule_score_preview = self.rule_analyzer(text, None).get("rule_score", 0)
            except Exception:
                rule_score_preview = 0

            # 비동기 Task 스케줄링
            text_task = asyncio.create_task(self._analyze_text_hybrid(text, rule_score_preview))
            url_task = None

            if has_url:
                original_url = urls[0]

                async def url_track():
                    traced = await trace_url(original_url)
                    res = await self.url_analyzer.scan_url(traced)
                    return traced, res

                url_task = asyncio.create_task(url_track())

            # 병렬 실행 공정
            if url_task:
                (
                    (text_analysis, naive_bayes_score, llm_available),
                    (traced_url, hybrid_res),
                ) = await asyncio.gather(text_task, url_task)
            else:
                (
                    text_analysis,
                    naive_bayes_score,
                    llm_available,
                ) = await text_task

                traced_url = None
                hybrid_res = {
                    "is_malicious": False,
                    "url_risk_score": 0.0,
                    "source": "Pre-Processing-Filter",
                    "available": False,
                    "failed_providers": [],
                    "pending_providers": [],
                    "provider_error_codes": {},
                    "error_message": None,
                    "is_gsb_confirmed": False,
                    "is_vt_confirmed": False,
                }

            text_result = (
                text_analysis.get("result") or {}
                if isinstance(text_analysis, dict)
                else {}
            )
            llm_score = text_result.get("risk_score", 0)
            text_available = naive_bayes_score is not None or llm_available

            # 로컬 규칙 기반 트랙: 금융기관 DB 대조 + 금융 키워드 + 계좌/카드번호 패턴 + 도메인 룰(.ru 등)
            try:
                rule_result = self.rule_analyzer(text, traced_url)
            except Exception as exception:
                logger.error(
                    "[Analysis Service] 규칙 분석 중 오류 발생. error_type=%s",
                    type(exception).__name__,
                )
                rule_result = {
                    "rule_score": 0,
                    "has_malicious_domain_pattern": False,
                    "matched_rules": [],
                    "error_message": "RULE_ANALYSIS_FAILED",
                }

            rules_available = not bool(rule_result.get("error_message"))
            url_available = has_url and hybrid_res.get("available", False)

            if rule_result.get("has_malicious_domain_pattern", False):
                hybrid_res["is_malicious"] = True
                hybrid_res["url_risk_score"] = max(
                    hybrid_res.get("url_risk_score", 0.0),
                    0.75,
                )

            # 확정 악성 판정 소스 3종 중 하나라도 해당하면 문맥 점수와 무관하게 HIGH 강제 오버라이드 대상.
            # (GSB 블랙리스트 등재 / VT 다수 엔진 합의 / 로컬 도메인 룰 매치 — 신뢰도 낮은 VT 소수 탐지는 제외)
            is_confirmed_malicious = (
                hybrid_res.get("is_gsb_confirmed", False)
                or hybrid_res.get("is_vt_confirmed", False)
                or rule_result.get("has_malicious_domain_pattern", False)
            )

            no_reliable_signal = (
                not text_available and not url_available and not rules_available
            )
            if no_reliable_signal:
                raise ValueError("No reliable analysis signal is available")

            # 3중 스코어링 최종 계산 (텍스트 트랙은 나이브 베이즈 + Gemini 하이브리드 결합 점수 사용,
            # URL 없으면 URL 트랙(30%)이 LLM/규칙 트랙으로 재배분됨)
            final_score, risk_grade, breakdown = RiskScoringEngine.calculate_score(
                llm_score=int(llm_score),
                is_url_malicious=hybrid_res.get("is_malicious", False),
                url_risk_score=hybrid_res.get("url_risk_score", 0.0),
                rule_score=rule_result.get("rule_score", 0),
                has_url=has_url,
                naive_bayes_score=naive_bayes_score,
                llm_available=llm_available,
                is_confirmed_malicious=is_confirmed_malicious,
                text_available=text_available,
                url_available=url_available,
                rules_available=rules_available,
            )

            # URL 부재 시 예외 방어 및 스켈레톤 분기벽 구축
            if has_url:
                real_url_analysis = {
                    "has_url": True,
                    "is_shortened": original_url != traced_url,
                    "origin_url": traced_url,
                    "original_url": original_url,
                    "is_url_malicious": hybrid_res.get("is_malicious", False),
                    "url_risk_score": hybrid_res.get("url_risk_score", 0.0),
                    "engine_source": hybrid_res.get("source", "Hybrid-Engine"),
                    "available": url_available,
                    "failed_providers": hybrid_res.get("failed_providers", []),
                    "pending_providers": hybrid_res.get("pending_providers", []),
                    "provider_error_codes": hybrid_res.get(
                        "provider_error_codes",
                        {},
                    ),
                    "error_message": hybrid_res.get("error_message"),
                }
            else:
                real_url_analysis = None

            return SmishingAnalysisResponse(
                status="SUCCESS",
                message="3중 가중치 결합 스미싱 통합 분석이 완료되었습니다.",
                final_score=final_score,
                risk_grade=risk_grade,
                contribution_breakdown=breakdown,
                text_analysis=text_analysis,
                url_analysis=real_url_analysis,
                rule_analysis=rule_result,
            )
        except Exception as exception:
            logger.error(
                "파이프라인 오류. error_type=%s",
                type(exception).__name__,
            )
            # 파이프라인이 통째로 죽어 어떤 트랙도 실행되지 못한 경우, final_score=0/LOW를
            # 반환하면 "분석 실패"가 "안전 확인됨"으로 읽혀 fail-open이 된다 (텍스트 트랙
            # 양쪽 엔진이 동시에 실패한 경우를 막는 BOTH_ENGINES_UNAVAILABLE_FALLBACK_SCORE와
            # 같은 이유). status="ERROR"만 보고 걸러내지 않는 소비자를 위해 등급/점수 자체를
            # 최소 MEDIUM으로 강제한다.
            return SmishingAnalysisResponse(
                status="ERROR",
                message="분석 파이프라인 처리 중 오류가 발생했습니다.",
                final_score=RiskScoringEngine.PIPELINE_FAILURE_FALLBACK_SCORE,
                risk_grade=RiskGrade.MEDIUM,
                contribution_breakdown=ContributionBreakdown(
                    llm=0, hybrid_url=0, rules=0
                ),
                text_analysis=None,
                url_analysis=None,
                rule_analysis=None,
            )

    async def scan_message_text(self, message: str) -> UrlAnalysisResponse:
        urls = extract_urls(message)
        if not urls:
            return UrlAnalysisResponse(
                has_url=False,
                original_url=None,
                traced_url=None,
                is_url_malicious=False,
                url_risk_score=0.0,
                engine_source="Pre-Processing-Filter",
            )
        traced_url = await trace_url(urls[0])
        res = await self.url_analyzer.scan_url(traced_url)
        return UrlAnalysisResponse(
            has_url=True,
            original_url=urls[0],
            traced_url=traced_url,
            is_url_malicious=res["is_malicious"],
            url_risk_score=res["url_risk_score"],
            engine_source=res["source"],
            error_message=res["error_message"],
        )
