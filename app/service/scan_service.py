import logging
import asyncio
from app.dto.schemas import URLScanRequest, URLScanResponse
from app.utils.url_tracker import extract_urls, trace_url

from app.service.security.mock_provider import is_mock_enabled, get_mock_security_data
from app.service.security.virustotal import check_virus_total
from app.service.security.google_safe_browsing import check_google_safe_browsing

logger = logging.getLogger(__name__)

async def process_url_scan_pipeline(request: URLScanRequest) -> URLScanResponse:
  
    try:
        # 본문 텍스트에서 URL 정규식 추출
        urls = extract_urls(request.text)
        
        #  URL이 없다면 안전한 클린 응답 리턴
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

        if is_mock_enabled():
            logger.info(f"[MOCK MODE] 시연용 가짜 보안 데이터를 로드합니다. Target: {traced_url}")
            mock_data = get_mock_security_data(traced_url)
            
            # 가짜 데이터 구조에서 위험 정보 취합
            gsb_malicious = mock_data.get("google_safe_browsing", False)
            vt_stats = mock_data.get("virustotal", {})
            vt_malicious_count = vt_stats.get("malicious", 0)
            
            # 위험도 및 악성 여부 자체 판정 룰 규칙 
            is_malicious = gsb_malicious or (vt_malicious_count > 0)
            
            # 스점 위험 점수 계산 (0.0 ~ 1.0 규격 매핑)
            risk_score = 0.0
            if is_malicious:
                risk_score = 0.95 if gsb_malicious else min(0.1 + (vt_malicious_count * 0.1), 1.0)
            else:
                risk_score = 0.0

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
        
        # VirusTotal과 Google Safe Browsing 동시 실행
        vt_task = asyncio.create_task(check_virus_total(traced_url))
        gsb_task = asyncio.create_task(check_google_safe_browsing(traced_url))
        
        # 두 비동기 태스크가 모두 끝날 때까지 대기
        vt_result, gsb_result = await asyncio.gather(vt_task, gsb_task)
        
        # [최종 의사결정 하이브리드 엔진 룰 규칙]
        # 1. Google Safe Browsing에서 차단되었거나, 
        # 2. VirusTotal 분석 엔진 중 3개 이상이 악성(malicious)이라고 판정한 경우 최종 악성으로 확정
        vt_malicious_count = vt_result.get("malicious", 0)
        is_malicious = gsb_result or (vt_malicious_count >= 3)
        
        # 위험도 산정 알고리즘 
        if gsb_result:
            risk_score = 0.95  
        elif vt_malicious_count > 0:
            # 탐지된 백신 엔진 개수에 비례하여 가중치 계산 
            risk_score = min(0.2 + (vt_malicious_count * 0.08), 1.0)
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
            original_url=request.text[:20] + "...",
            traced_url=None,
            is_url_malicious=False,
            url_risk_score=0.0,
            engine_source="Hybrid-Engine-Failure",
            error_message=str(e)
        )