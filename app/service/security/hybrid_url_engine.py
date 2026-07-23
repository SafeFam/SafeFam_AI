import os
import logging
import asyncio
from app.service.security.virustotal import VirusTotalEngine
from app.service.security.google_safe_browsing import GoogleSafeBrowsingEngine

logger = logging.getLogger(__name__)
MOCK_ENABLED = os.getenv("MOCK_SECURITY_API", "False").lower() in ("true", "1", "t")

# Google Safe Browsing(1차)과 VirusTotal(2차 백업)을 제어하는 하이브리드 URL 분석 코어 엔진
class HybridUrlEngine:

    def __init__(self):
        self.vt_engine = VirusTotalEngine()
        self.gsb_engine = GoogleSafeBrowsingEngine()

    async def scan_url(self, traced_url: str) -> dict:
        # 쉘 환경변수에 따른 MOCK 모드 분기 로직 정상화
        if MOCK_ENABLED:
            logger.info(f"[MOCK MODE] 하이브리드 URL 스캔 -> Target: {traced_url}")
            return {
                "is_malicious": True, 
                "url_risk_score": 0.85, 
                "source": "Hybrid-Engine (MOCK)", 
                "detected_count": 4, 
                "error_message": None
            }

        # PROD 운영 모드 가동 
        logger.info("[PROD MODE] 1차 방어선: Google Safe Browsing API 가동")
        error_logs = []

        try:
            gsb_result = await self.gsb_engine.scan_url(traced_url)
            is_gsb_blocked = gsb_result.get("is_malicious", False)
        except Exception as e:
            logger.error(f"GSB 통신 실패: {str(e)}")
            gsb_result = {"is_malicious": False}
            is_gsb_blocked = False
            error_logs.append(f"GSB Fail ({str(e)[:15]})")

        vt_result = {"is_malicious": False, "detected_count": 0}

        # GSB 악성 확정 시 VT 생략 (Quota 절약)
        if is_gsb_blocked:
            logger.info(" GSB 악성 판정으로 VirusTotal 호출 생략 (Quota 절약)")
            engine_source = "Hybrid-Engine (GSB)"
            risk_score = gsb_result.get("raw_score", 0.95)
            if risk_score > 1.0: 
                risk_score /= 100.0
        else:
            logger.info(" GSB 청정/불확실로 인한 2차 방어선 VirusTotal 백업 가동")
            engine_source = "Hybrid-Engine (GSB+VT)"
            try:
                vt_result = await asyncio.wait_for(self.vt_engine.scan_url(traced_url), timeout=4.0)
            except Exception as e:
                logger.error(f"VirusTotal 통신 실패: {str(e)}")
                vt_result = {"is_malicious": False, "detected_count": 0}
                error_logs.append(f"VT Fail ({str(e)[:15]})")

            vt_malicious_count = vt_result.get("detected_count", 0)
            if vt_malicious_count > 0:
                base_score = vt_result.get("raw_score", 0.0)
                if base_score > 1.0: 
                    base_score /= 100.0
                risk_score = max(base_score, min(0.1 + (vt_malicious_count * 0.15), 0.95))
            else:
                risk_score = 0.0
        
        is_final_malicious = is_gsb_blocked or vt_result.get("is_malicious", False) or vt_result.get("detected_count", 0) >= 1
        combined_error = " | ".join(error_logs) if error_logs else None

        return {
            "is_malicious": is_final_malicious,
            "url_risk_score": round(risk_score, 2),
            "source": engine_source,
            "error_message": combined_error
        }