import logging
import asyncio
from typing import Optional
from app.dto.schemas import SmishingAnalysisResponse, URLScanResponse, RiskGrade, ContributionBreakdown
from app.utils.url_tracker import extract_urls, trace_url
from app.utils.scoring_engine import ScoringEngine
from app.service.security.gemini_text_analyzer import analyze_text_with_gemini
from app.service.security.naive_bayes_text_analyzer import analyze_text_with_naive_bayes
from app.service.security.rule_based_analyzer import analyze_text_with_rules
from app.service.security.hybrid_url_engine import HybridUrlEngine

logger = logging.getLogger(__name__)

class ScanService:
    def __init__(self):
        self.hybrid_url_engine = HybridUrlEngine()

    # 1차 나이브 베이즈 선별 후, SAFE 기준 미만(의심)일 때만 Gemini 2차 검증을 호출하는 하이브리드 텍스트 트랙
    # 반환값: (text_analysis 응답용 dict, 나이브 베이즈 점수(미수행 시 None), Gemini 2차 검증 정상 수행 여부)
    async def _analyze_text_hybrid(self, text: str) -> tuple[dict, Optional[int], bool]:
        nb_result = await analyze_text_with_naive_bayes(text)
        nb_grade = nb_result.get("result", {}).get("grade")
        nb_score = nb_result.get("result", {}).get("risk_score", 0)
        nb_available = nb_result.get("is_available", False)

        # 모델이 정상 로드되어 SAFE로 판정한 경우에만 Gemini 호출 스킵.
        # 모델 로드 실패(UNKNOWN)는 안전하게 Gemini 2차 검증으로 폴백(fail-safe).
        if nb_available and nb_grade == "SAFE":
            logger.info("[Hybrid] 나이브 베이즈 SAFE 판정 - Gemini 2차 검증 스킵")
            return nb_result, nb_score, False

        text_analysis = await analyze_text_with_gemini(text)
        text_analysis["stage1_naive_bayes"] = nb_result.get("result")

        llm_available = text_analysis.get("result", {}).get("grade") != "UNKNOWN"
        return text_analysis, (nb_score if nb_available else None), llm_available

    # 텍스트 트랙과 URL 트랙을 병렬 조립하고 스코어링을 매핑
    async def analyze_pipeline(self, text: str) -> SmishingAnalysisResponse:

        try:
            urls = extract_urls(text)
            has_url = len(urls) > 0

            # 비동기 Task 스케줄링
            text_task = asyncio.create_task(self._analyze_text_hybrid(text))
            url_task = None

            if has_url:
                original_url = urls[0]
                async def url_track():
                    traced = await trace_url(original_url)
                    res = await self.hybrid_url_engine.scan_url(traced)
                    return traced, res
                url_task = asyncio.create_task(url_track())

            # 병렬 실행 공정
            if url_task:
                (text_analysis, naive_bayes_score, llm_available), (traced_url, hybrid_res) = await asyncio.gather(text_task, url_task)
            else:
                text_analysis, naive_bayes_score, llm_available = await text_task
                traced_url, hybrid_res = None, {"is_malicious": False, "url_risk_score": 0.0, "source": "Pre-Processing-Filter", "error_message": None, "is_gsb_confirmed": False, "is_vt_confirmed": False}

            llm_score = text_analysis.get("result", {}).get("risk_score", 0) if isinstance(text_analysis, dict) else 0

            # 로컬 규칙 기반 트랙: 금융기관 DB 대조 + 금융 키워드 + 계좌/카드번호 패턴 + 도메인 룰(.ru 등)
            rule_result = analyze_text_with_rules(text, traced_url)
            if rule_result["has_malicious_domain_pattern"]:
                hybrid_res["is_malicious"] = True
                hybrid_res["url_risk_score"] = max(hybrid_res["url_risk_score"], 0.75)

            # 확정 악성 판정 소스 3종 중 하나라도 해당하면 문맥 점수와 무관하게 HIGH 강제 오버라이드 대상.
            # (GSB 블랙리스트 등재 / VT 다수 엔진 합의 / 로컬 도메인 룰 매치 — 신뢰도 낮은 VT 소수 탐지는 제외)
            is_confirmed_malicious = (
                hybrid_res.get("is_gsb_confirmed", False)
                or hybrid_res.get("is_vt_confirmed", False)
                or rule_result["has_malicious_domain_pattern"]
            )

            # 3중 스코어링 최종 계산 (텍스트 트랙은 나이브 베이즈 + Gemini 하이브리드 결합 점수 사용,
            # URL 없으면 URL 트랙(30%)이 LLM/규칙 트랙으로 재배분됨)
            final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
                llm_score=int(llm_score),
                is_url_malicious=hybrid_res["is_malicious"],
                url_risk_score=hybrid_res["url_risk_score"],
                rule_score=rule_result["rule_score"],
                has_url=has_url,
                naive_bayes_score=naive_bayes_score,
                llm_available=llm_available,
                is_confirmed_malicious=is_confirmed_malicious
            )

            # URL 부재 시 예외 방어 및 스켈레톤 분기벽 구축
            if has_url:
                real_url_analysis = {
                    "has_url": True,
                    "is_shortened": original_url != traced_url,
                    "origin_url": traced_url,
                    "original_url": original_url,
                    "is_url_malicious": hybrid_res["is_malicious"],
                    "url_risk_score": hybrid_res["url_risk_score"],
                    "engine_source": hybrid_res["source"],
                    "error_message": hybrid_res["error_message"]
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
                rule_analysis=rule_result
            )
        except Exception as e:
            logger.error(f"파이프라인 에러: {str(e)}")
            return SmishingAnalysisResponse(
                status="ERROR",
                message=str(e),
                final_score=0,
                risk_grade=RiskGrade.LOW,
                contribution_breakdown=ContributionBreakdown(llm=0, hybrid_url=0, rules=0),
                text_analysis=None,
                url_analysis=None,
                rule_analysis=None
            )

    async def scan_message_text(self, message: str) -> URLScanResponse:
        urls = extract_urls(message)
        if not urls: 
            return URLScanResponse(
                has_url=False, 
                original_url=None, 
                traced_url=None, 
                is_url_malicious=False, 
                url_risk_score=0.0, 
                engine_source="Pre-Processing-Filter"
            )
        traced_url = await trace_url(urls[0])
        res = await self.hybrid_url_engine.scan_url(traced_url)
        return URLScanResponse(
            has_url=True, 
            original_url=urls[0], 
            traced_url=traced_url, 
            is_url_malicious=res["is_malicious"], 
            url_risk_score=res["url_risk_score"], 
            engine_source=res["source"], 
            error_message=res["error_message"]
        )