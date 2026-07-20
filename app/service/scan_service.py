import logging
import asyncio
from app.dto.schemas import SmishingAnalysisResponse, URLScanResponse, RiskGrade, ContributionBreakdown
from app.utils.url_tracker import extract_urls, trace_url
from app.utils.scoring_engine import ScoringEngine
from app.service.security.gemini_text_analyzer import analyze_text_with_gemini
from app.service.security.hybrid_url_engine import HybridUrlEngine 

logger = logging.getLogger(__name__)

class ScanService:
    def __init__(self):
        self.hybrid_url_engine = HybridUrlEngine()  

    # 텍스트 트랙과 URL 트랙을 병렬 조립하고 스코어링을 매핑
    async def analyze_pipeline(self, message: str) -> SmishingAnalysisResponse:
      
        try:
            urls = extract_urls(message)
            has_url = len(urls) > 0

            # 비동기 Task 스케줄링
            text_task = asyncio.create_task(analyze_text_with_gemini(message))
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
                text_analysis, (traced_url, hybrid_res) = await asyncio.gather(text_task, url_task)
            else:
                text_analysis = await text_task
                traced_url, hybrid_res = None, {"is_malicious": False, "url_risk_score": 0.0, "source": "Pre-Processing-Filter", "error_message": None}

            llm_score = text_analysis.get("result", {}).get("risk_score", 0) if isinstance(text_analysis, dict) else 0

            # 로컬 규칙 패널티 가드
            has_rule_violation = False
            if has_url and traced_url and (".ru" in traced_url or "testsafebrowsing" in traced_url):
                has_rule_violation = True
                hybrid_res["is_malicious"] = True
                hybrid_res["url_risk_score"] = max(hybrid_res["url_risk_score"], 0.75)

            # 3중 스코어링 최종 계산
            final_score, risk_grade, breakdown = ScoringEngine.calculate_score(
                llm_score=int(llm_score),
                is_url_malicious=hybrid_res["is_malicious"],
                url_risk_score=hybrid_res["url_risk_score"],
                has_rule_violation=has_rule_violation
            )

            return SmishingAnalysisResponse(
                status="SUCCESS",
                message="3중 가중치 결합 스미싱 통합 분석이 완료되었습니다.",
                final_score=final_score,
                risk_grade=risk_grade,
                contribution_breakdown=breakdown,
                text_analysis=text_analysis,
                url_analysis={
                    "has_url": has_url,
                    "is_shortened": original_url != traced_url if has_url else False,
                    "origin_url": traced_url,
                    "original_url": original_url if has_url else None,
                    "is_url_malicious": hybrid_res["is_malicious"],
                    "url_risk_score": hybrid_res["url_risk_score"],
                    "engine_source": hybrid_res["source"],
                    "error_message": hybrid_res["error_message"]
                }
            )
        except Exception as e:
            logger.error(f"파이프라인 에러: {str(e)}")
            return SmishingAnalysisResponse(status="ERROR", message=str(e), final_score=0, risk_grade=RiskGrade.LOW, contribution_breakdown=ContributionBreakdown(llm=0, hybrid_url=0, rules=0))

    async def scan_message_text(self, message: str) -> URLScanResponse:
        urls = extract_urls(message)
        if not urls: return URLScanResponse(has_url=False, original_url=None, traced_url=None, is_url_malicious=False, url_risk_score=0.0, engine_source="Pre-Processing-Filter")
        traced_url = await trace_url(urls[0])
        res = await self.hybrid_url_engine.scan_url(traced_url)
        return URLScanResponse(has_url=True, original_url=urls[0], traced_url=traced_url, is_url_malicious=res["is_malicious"], url_risk_score=res["url_risk_score"], engine_source=res["source"], error_message=res["error_message"])