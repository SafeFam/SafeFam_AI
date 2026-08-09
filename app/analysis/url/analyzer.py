import logging
from typing import ClassVar

from app.analysis.ports import UrlSecurityProvider
from app.core.config import settings
from app.infrastructure.google_safe_browsing.client import (
    GoogleSafeBrowsingClient,
)
from app.infrastructure.virustotal.client import (
    VirusTotalClient,
)

logger = logging.getLogger(__name__)

MOCK_ENABLED = settings.MOCK_SECURITY_API


# Google Safe Browsing(1차)과 VirusTotal(2차 백업)을 제어하는 하이브리드 URL 분석 코어 엔진
class HybridUrlAnalyzer:
    # VT 탐지 엔진 수가 이 이상이면 "다수 백신사 합의"로 보고 GSB와 동급으로 확정 취급.
    # (VT는 개별 오탐 벤더가 섞여있어 GSB 블랙리스트 등재만큼 신뢰도가 높진 않지만,
    #  다수 엔진이 동시에 일치하면 우연한 오탐일 가능성이 낮아짐)
    VT_CONFIRMED_ENGINE_THRESHOLD = 5

    AVAILABLE_STATUSES: ClassVar[frozenset[str]] = frozenset({"safe", "completed"})

    def __init__(
        self,
        vt_client: UrlSecurityProvider | None = None,
        gsb_client: UrlSecurityProvider | None = None,
    ):
        self.vt_client = vt_client or VirusTotalClient()
        self.gsb_client = gsb_client or GoogleSafeBrowsingClient()

    @classmethod
    def _is_available(cls, result: dict) -> bool:
        """분석 결과 사용 가능한 상태"""
        return result.get("status") in cls.AVAILABLE_STATUSES

    @staticmethod
    def _is_unavailable(result: dict) -> bool:
        """분석 결과 불가 상태"""
        return result.get("status") == "unavailable"

    async def scan_url(self, traced_url: str) -> dict:
        # 쉘 환경변수에 따른 MOCK 모드 분기 로직 정상화
        if MOCK_ENABLED:
            logger.info("[MOCK MODE] 하이브리드 URL 스캔 시작")
            return {
                "is_malicious": True,
                "url_risk_score": 0.85,
                "source": "Hybrid-Engine (MOCK)",
                "detected_count": 4,
                "available": True,
                "failed_providers": [],
                "pending_providers": [],
                "provider_error_codes": {},
                "error_message": None,
                "is_gsb_confirmed": True,
                "is_vt_confirmed": False,
            }

        # PROD 운영 모드 가동
        logger.info("[PROD MODE] Google Safe Browsing 분석 시작")

        failed_providers: list[str] = []
        pending_providers: list[str] = []
        error_messages: list[str] = []
        provider_error_codes: dict[str, str] = {}

        # 1차 분석: GSB
        gsb_result = await self._scan_gsb(traced_url)

        gsb_available = self._is_available(gsb_result)
        is_gsb_blocked = gsb_available and gsb_result.get("is_malicious", False)

        if self._is_unavailable(gsb_result):
            failed_providers.append("GSB")

            error_code = gsb_result.get(
                "error_code",
                "UNKNOWN",
            )
            provider_error_codes["GSB"] = error_code
            error_messages.append(f"GSB unavailable ({error_code})")

        # GSB에서 악성 URL을 확정한 경우 VirusTotal 호출 X
        if is_gsb_blocked:
            logger.info(" GSB 악성 판정으로 VirusTotal 호출 생략 (Quota 절약)")
            risk_score = gsb_result.get("raw_score", 0.95)
            if risk_score > 1.0:
                risk_score /= 100.0

            return {
                "is_malicious": True,
                "url_risk_score": round(
                    risk_score,
                    2,
                ),
                "source": "Hybrid-Engine (GSB)",
                "detected_count": gsb_result.get(
                    "detected_count",
                    1,
                ),
                "available": True,
                "failed_providers": failed_providers,
                "pending_providers": pending_providers,
                "provider_error_codes": provider_error_codes,
                "error_message": (
                    " | ".join(error_messages) if error_messages else None
                ),
                "is_gsb_confirmed": True,
                "is_vt_confirmed": False,
            }

        # 2차 분석: VirusTotal 백업 분석
        logger.info("[Hybrid URL] VirusTotal 백업 분석 시작")

        vt_result = await self._scan_virustotal(traced_url)

        vt_status = vt_result.get("status")
        vt_available = self._is_available(vt_result)

        if self._is_unavailable(vt_result):
            failed_providers.append("VIRUSTOTAL")

            error_code = vt_result.get(
                "error_code",
                "UNKNOWN",
            )
            provider_error_codes["VIRUSTOTAL"] = error_code
            error_messages.append(f"VirusTotal unavailable ({error_code})")

        elif vt_status == "scanning":
            pending_providers.append("VIRUSTOTAL")

        vt_malicious_count = vt_result.get("detected_count", 0) if vt_available else 0

        # VT 분석 결과를 바탕으로 최종 위험도 점수 재계산
        if vt_available and vt_malicious_count > 0:
            base_score = vt_result.get(
                "raw_score",
                0.0,
            )

            if base_score > 1.0:
                base_score /= 100.0

            risk_score = max(
                base_score,
                min(
                    0.1 + vt_malicious_count * 0.15,
                    0.95,
                ),
            )
        else:
            risk_score = 0.0

        is_vt_malicious = vt_available and vt_result.get(
            "is_malicious",
            False,
        )

        is_vt_confirmed = (
            vt_available and vt_malicious_count >= self.VT_CONFIRMED_ENGINE_THRESHOLD
        )

        is_final_malicious = is_gsb_blocked or is_vt_malicious

        # 두 제공자 중 하나라도 정상 결과를 반환하면 URL트랙은 사용 가능으로 판정
        available = gsb_available or vt_available

        if not available and not error_messages:
            error_messages.append("No URL provider returned a completed result")

        return {
            "is_malicious": is_final_malicious,
            "url_risk_score": round(risk_score, 2),
            "source": "Hybrid-Engine (GSB+VT)",
            "detected_count": vt_malicious_count,
            "available": available,
            "failed_providers": failed_providers,
            "pending_providers": pending_providers,
            "provider_error_codes": provider_error_codes,
            "error_message": (" | ".join(error_messages) if error_messages else None),
            "is_gsb_confirmed": is_gsb_blocked,
            "is_vt_confirmed": is_vt_confirmed,
        }

    async def _scan_gsb(
        self,
        traced_url: str,
    ) -> dict:
        """GSB 클라이언트를 호출하고 예외 발생시 예외 응답 반환"""
        try:
            return await self.gsb_client.scan_url(traced_url)

        except Exception as exception:
            logger.error(
                "[Hybrid URL] GSB 호출 중 예외 발생. error_type=%s",
                type(exception).__name__,
            )
            return {
                "is_malicious": False,
                "raw_score": 0.0,
                "detected_count": 0,
                "status": "unavailable",
                "error_code": "UNEXPECTED_ERROR",
            }

    async def _scan_virustotal(
        self,
        traced_url: str,
    ) -> dict:
        """VirusTotal 클라이언트를 호출하고 실패 결과로 변환"""
        try:
            return await self.vt_client.scan_url(traced_url)

        except Exception as exception:
            logger.error(
                "[Hybrid URL] VirusTotal 호출 중 예외 발생. error_type=%s",
                type(exception).__name__,
            )
            return {
                "is_malicious": False,
                "raw_score": 0.0,
                "detected_count": 0,
                "status": "unavailable",
                "error_code": "UNEXPECTED_ERROR",
            }
