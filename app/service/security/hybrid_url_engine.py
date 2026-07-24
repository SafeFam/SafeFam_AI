import os
import logging
import asyncio
from app.service.security.virustotal import VirusTotalEngine
from app.service.security.google_safe_browsing import GoogleSafeBrowsingEngine

logger = logging.getLogger(__name__)
MOCK_ENABLED = os.getenv("MOCK_SECURITY_API", "False").lower() in ("true", "1", "t")

# Google Safe Browsing(1차)과 VirusTotal(2차 백업)을 제어하는 하이브리드 URL 분석 코어 엔진
class HybridUrlEngine:

    # VT 탐지 엔진 수가 이 이상이면 "다수 백신사 합의"로 보고 GSB와 동급으로 확정 취급.
    # (VT는 개별 오탐 벤더가 섞여있어 GSB 블랙리스트 등재만큼 신뢰도가 높진 않지만,
    #  다수 엔진이 동시에 일치하면 우연한 오탐일 가능성이 낮아짐)
    VT_CONFIRMED_ENGINE_THRESHOLD = 5

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
                "error_message": None,
                "is_gsb_confirmed": True,
                "is_vt_confirmed": False
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
        is_vt_confirmed = False

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

            # 다수 백신 엔진이 동시에 악성으로 합의한 경우만 GSB급 확정 신호로 승격
            is_vt_confirmed = vt_malicious_count >= self.VT_CONFIRMED_ENGINE_THRESHOLD

        is_final_malicious = is_gsb_blocked or vt_result.get("is_malicious", False) or vt_result.get("detected_count", 0) >= 1
        combined_error = " | ".join(error_logs) if error_logs else None

        return {
            "is_malicious": is_final_malicious,
            "url_risk_score": round(risk_score, 2),
            "source": engine_source,
            "error_message": combined_error,
            # Google Safe Browsing 블랙리스트에 실제로 등재되어 확인된 경우만 True.
            "is_gsb_confirmed": is_gsb_blocked,
            # VT 탐지 엔진 수가 임계치 이상인 "다수 합의" 케이스만 True (스코어링 엔진의 확정 악성 오버라이드 트리거용)
            "is_vt_confirmed": is_vt_confirmed
        }