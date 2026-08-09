import logging

import httpx

from app.core.config import settings
from app.infrastructure.http_retry import request_with_retry

logger = logging.getLogger(__name__)


class GoogleSafeBrowsingClient:
    """Google Safe Browsing API를 사용하여 URL의 악성 여부를 검사"""

    def __init__(self):
        self.api_key = settings.GOOGLE_SAFE_BROWSING_API_KEY
        self.api_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

    @staticmethod
    def _safe_result() -> dict:
        """GSB가 실제로 URL을 조회한 뒤 안전하다고 판단한 결과"""
        return {
            # 실제 안전 판정
            "is_malicious": False,
            "raw_score": 0.0,
            "detected_count": 0,
            "status": "safe",
            "error_code": None,
        }

    @staticmethod
    def _unavailable_result(error_code: str) -> dict:
        """API 장애로 URL의 안전 여부를 판단하지 못한 결과"""
        # API 장애로 판정 불가
        return {
            "is_malicious": False,
            "raw_score": 0.0,
            "detected_count": 0,
            "status": "unavailable",
            "error_code": error_code,
        }

    async def scan_url(self, url: str) -> dict:
        """입력받은 URL을 GSB API로 검사 후 분석 결과 반환"""
        if not self.api_key:
            logger.warning(
                "[Google Safe Browsing] API Key가 누락되어 URL을 분석할 수 없습니다."
            )
            return self._unavailable_result("MISSING_API_KEY")

        # GSB API 요청 페이로드 구성
        payload = {
            "client": {
                "clientId": "safefam-ai-backend",
                "clientVersion": "1.0.0",
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [
                    {"url": url},
                ],
            },
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await request_with_retry(
                    lambda: client.post(
                        self.api_url,
                        params={"key": self.api_key},
                        json=payload,
                        timeout=settings.GSB_TIMEOUT_SECONDS,
                    ),
                    max_retries=(settings.EXTERNAL_API_MAX_RETRIES),
                    operation_name="Google Safe Browsing",
                )
                response.raise_for_status()

                result = response.json()
                matches = result.get("matches", [])

                # 매칭되는 위험 요소가 있는 경우 악성 URL로 처리
                if matches:
                    logger.warning("[Google Safe Browsing] 악성 URL 감지됨")
                    return {
                        "is_malicious": True,
                        "raw_score": 0.95,
                        "detected_count": len(matches),
                        "status": "completed",
                        "error_code": None,
                    }

                logger.info("[Google Safe Browsing] URL 분석 완료")
                return self._safe_result()

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code

                logger.error(
                    "[Google Safe Browsing] API 에러 (%s)",
                    status_code,
                )

                if status_code == 429:
                    return self._unavailable_result("RATE_LIMITED")

                return self._unavailable_result(f"HTTP_{status_code}")

            except httpx.TimeoutException:
                logger.error("[Google Safe Browsing] API 요청 타임아웃 발생")
                return self._unavailable_result("TIMEOUT")

            except httpx.RequestError as exc:
                logger.error(
                    "[Google Safe Browsing] 네트워크 오류: %s",
                    type(exc).__name__,
                )
                return self._unavailable_result("NETWORK_ERROR")

            except Exception as exception:
                logger.error(
                    "[Google Safe Browsing] 연동 중 비정상 오류 발생. error_type=%s",
                    type(exception).__name__,
                )
                return self._unavailable_result("UNEXPECTED_ERROR")
