import base64
import logging
import httpx
from app.core.config import settings
from app.infrastructure.http_retry import request_with_retry

logger = logging.getLogger(__name__)

class VirusTotalClient:
    """VirusTotal v3 API를 통해 URL의 악성 여부를 검사하고 스캔 요청"""
    def __init__(self):
        self.api_key = settings.VIRUSTOTAL_API_KEY
        self.base_url = "https://www.virustotal.com/api/v3"
        self.headers = {
            "x-apikey": self.api_key or "",
            "accept": "application/json",
        }

    @staticmethod
    def _safe_result() -> dict:
        """VirusTotal이 실제로 URL을 조회한 뒤 안전하다고 판단한 결과"""
        return {
            "is_malicious": False,
            "raw_score": 0.0,
            "detected_count": 0,
            "total_engines": 0,
            "status": "safe",
            "error_code": None,
        }

    @staticmethod
    def _unavailable_result(error_code: str) -> dict:
        """API 장애로 URL의 안전 여부를 판단하지 못한 결과"""
        return {
            "is_malicious": False,
            "raw_score": 0.0,
            "detected_count": 0,
            "total_engines": 0,
            "status": "unavailable",
            "error_code": error_code,
        }

    @staticmethod
    def _scanning_result() -> dict:
        """신규 분석을 요청했지만 결과가 아직 준비되지 않은 상태"""
        return {
            "is_malicious": False,
            "raw_score": 0.0,
            "detected_count": 0,
            "total_engines": 0,
            "status": "scanning",
            "error_code": None,
        }

    def _get_url_id(self, url: str) -> str:
        """URL을 Base64 URL-safe 식별자로 변환"""
        b64_bytes = base64.urlsafe_b64encode(
            url.encode("utf-8")
        )
        return b64_bytes.decode("utf-8").rstrip("=")

    async def scan_url(self, url: str) -> dict:
        """VT에 등록된 URL분석 보고서 조회하고 악성 위험도를 계산하여 반환"""
        if not self.api_key:
            logger.warning(
                "[VirusTotal] API Key가 누락되어 "
                "URL을 분석할 수 없습니다."
            )
            return self._unavailable_result(
                "MISSING_API_KEY"
            )

        url_id = self._get_url_id(url)
        report_url = f"{self.base_url}/urls/{url_id}"

        async with httpx.AsyncClient() as client:
            try:
                response = await request_with_retry(
                    lambda: client.get(
                        report_url,
                        headers=self.headers,
                        timeout=settings.VIRUSTOTAL_TIMEOUT_SECONDS,
                    ),
                    max_retries=settings.EXTERNAL_API_MAX_RETRIES,
                    operation_name="VirusTotal report",
                )

                # 기존 분석 보고서가 존재하지 않는 경우 신규 스캔 요청
                if response.status_code == 404:
                    return await self._request_new_scan(
                        client=client,
                        url=url,
                    )

                if response.status_code == 429:
                    logger.error(
                        "[VirusTotal] API 호출 한도 초과"
                    )
                    return self._unavailable_result(
                        "RATE_LIMITED"
                    )

                response.raise_for_status()

                report_data = response.json()
                stats = (
                    report_data
                    .get("data", {})
                    .get("attributes", {})
                    .get("last_analysis_stats", {})
                )

                malicious = stats.get("malicious", 0)
                suspicious = stats.get("suspicious", 0)
                total_engines = (
                    sum(stats.values()) if stats else 0
                )

                logger.info(
                    "[VirusTotal] 분석 완료 - "
                    "악성: %s, 의심: %s, 전체 엔진: %s",
                    malicious,
                    suspicious,
                    total_engines,
                )

                # 임계값 기준
                is_malicious = (
                    malicious >= 3
                    or malicious + suspicious >= 5
                )

                # 전체 분석 엔진 대비 위험 비율 기반 가중치 점수 계산
                if total_engines > 0:
                    malicious_ratio = (
                        malicious / total_engines
                    )
                    suspicious_ratio = (
                        suspicious / total_engines
                    )
                    raw_score = min(
                        malicious_ratio
                        + suspicious_ratio * 0.5,
                        1.0,
                    )
                else:
                    raw_score = 0.0

                status = (
                    "completed"
                    if is_malicious
                    else "safe"
                )

                return {
                    "is_malicious": is_malicious,
                    "raw_score": round(raw_score, 2),
                    "detected_count": malicious,
                    "total_engines": total_engines,
                    "status": status,
                    "error_code": None,
                }

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code

                logger.error(
                    "[VirusTotal] API 에러. status_code=%s",
                    status_code,
                )

                if status_code == 429:
                    return self._unavailable_result(
                        "RATE_LIMITED"
                    )

                return self._unavailable_result(
                    f"HTTP_{status_code}"
                )

            except httpx.TimeoutException:
                logger.error(
                    "[VirusTotal] API 요청 타임아웃 발생"
                )
                return self._unavailable_result("TIMEOUT")

            except httpx.RequestError as exc:
                logger.error(
                    "[VirusTotal] 네트워크 오류. error_type=%s",
                    type(exc).__name__,
                )
                return self._unavailable_result(
                    "NETWORK_ERROR"
                )

            except Exception as exception:
                logger.error(
                    "[VirusTotal] 연동 중 비정상 오류 발생. error_type=%s",
                    type(exception).__name__,
                )
                return self._unavailable_result(
                    "UNEXPECTED_ERROR"
                )

    async def _request_new_scan(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
    ) -> dict:
        """기존 보고서가 없는 URL에 대해 VT에 신규 스캔 분석 요청"""
        logger.info(
            "[VirusTotal] 기존 보고서 없음. "
            "신규 스캔 요청 시작",
        )

        scan_url = f"{self.base_url}/urls"
        scan_response = await request_with_retry(
            lambda: client.post(
                scan_url,
                headers=self.headers,
                data={"url": url},
                timeout=settings.VIRUSTOTAL_TIMEOUT_SECONDS,
            ),
            max_retries=settings.EXTERNAL_API_MAX_RETRIES,
            operation_name="VirusTotal scan",
        )

        if scan_response.status_code == 429:
            logger.error(
                "[VirusTotal] 신규 스캔 요청 한도 초과"
            )
            return self._unavailable_result(
                "RATE_LIMITED"
            )

        scan_response.raise_for_status()

        logger.info(
            "[VirusTotal] 신규 스캔 요청 완료",
        )
        return self._scanning_result()
