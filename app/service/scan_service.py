import os
import logging
import asyncio
from app.dto.schemas import URLScanResponse
from app.utils.url_tracker import extract_urls, trace_url

from app.service.security.virustotal import VirusTotalEngine
from app.service.security.google_safe_browsing import GoogleSafeBrowsingEngine

logger = logging.getLogger(__name__)

MOCK_ENABLED = os.getenv("MOCK_SECURITY_API", "False").lower() in ("true", "1", "t")

class ScanService:
    def __init__(self):
        self.vt_engine = VirusTotalEngine()
        self.gsb_engine = GoogleSafeBrowsingEngine()

    async def scan_message_text(self, message: str) -> URLScanResponse:
        """
        주입된 텍스트 본문 내 URL을 추출/추적하고 하이브리드 엔진으로 정밀 스캔을 수행합니다.
        """
        try:
            # 본문 텍스트에서 URL 정규식 추출
            urls = extract_urls(message)
            
            # URL이 없다면 안전한 클린 응답 리턴
            if not urls:
                logger.info(" 본문 내에 추출된 URL이 없어 안전한 상태로 판정합니다.")
                return URLScanResponse(
                    has_url=False,
                    original_url=None,
                    traced_url=None,
                    is_url_malicious=False,
                    url_risk_score=0.0,
                    engine_source="Pre-Processing-Filter",
                    error_message=None
                )
                
            # URL이 존재하면 첫 번째 URL을 추출하여 단축 URL 리다이렉트 비동기 추적
            original_url = urls[0]
            logger.info(f" URL 추출 완료: {original_url} -> 리다이렉트 추적 시작")
            
            traced_url = await trace_url(original_url)
            logger.info(f" 최종 도달 URL 추적 완료: {traced_url}")

            if MOCK_ENABLED:
                logger.info(f"[MOCK MODE] 목 데이터를 로드합니다. Target: {traced_url}")
                
                gsb_malicious = False
                vt_malicious_count = 4  
                
                is_malicious = gsb_malicious or (vt_malicious_count > 0)
                risk_score = 0.95 if gsb_malicious else min(0.1 + (vt_malicious_count * 0.1), 1.0) if is_malicious else 0.0

                return URLScanResponse(
                    has_url=True,
                    original_url=original_url,
                    traced_url=traced_url,
                    is_url_malicious=is_malicious,
                    url_risk_score=risk_score,
                    engine_source="Hybrid-Engine (MOCK)",
                    error_message=None
                )

            logger.info(f"[PROD MODE] 실제 VT 및 GSB API를 비동기 병렬 호출합니다.")
            
            # 비동기 병렬 태스크 스케줄링
            vt_task = asyncio.create_task(self.vt_engine.scan_url(traced_url))
            gsb_task = asyncio.create_task(self.gsb_engine.scan_url(traced_url))
            
            # 두 비동기 태스크가 모두 끝날 때까지 대기
            vt_result, gsb_result = await asyncio.gather(vt_task, gsb_task)
            
            # 공통 dict 규격에서 안전하게 결과 파싱
            vt_malicious_count = vt_result.get("detected_count", 0)
            is_gsb_blocked = gsb_result.get("is_malicious", False)
            
            # 하이브리드 의사결정 규칙 적용
            is_malicious = is_gsb_blocked or (vt_malicious_count >= 3)
            
            # 위험도 산정 알고리즘 통합
            if is_gsb_blocked:
                risk_score = gsb_result.get("raw_score", 0.95)
            elif vt_malicious_count > 0:
                risk_score = vt_result.get("raw_score", 0.0)
            else:
                risk_score = 0.0

            return URLScanResponse(
                has_url=True,
                original_url=original_url,
                traced_url=traced_url,
                is_url_malicious=is_malicious,
                url_risk_score=round(risk_score, 2),
                engine_source="Hybrid-Engine (VT+GSB)",
                error_message=None
            )

        except Exception as e:
            logger.error(f" 파이프라인 수행 중 예외 발생: {str(e)}")
            return URLScanResponse(
                has_url=True,
                original_url=message[:20] + "...",
                traced_url=None,
                is_url_malicious=False,
                url_risk_score=0.0,
                engine_source="Hybrid-Engine-Failure",
                error_message=str(e)
            )