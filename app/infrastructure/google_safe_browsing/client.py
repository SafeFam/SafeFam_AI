import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

class GoogleSafeBrowsingClient:
    def __init__(self):
        self.api_key = settings.GOOGLE_SAFE_BROWSING_API_KEY
        self.api_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={self.api_key}"

    # Google Safe Browsing API를 사용하여 URL의 실시간 악성 블랙리스트 등재 여부를 검사
    async def scan_url(self, url: str) -> dict:
        
        # 공통 인터페이스 리턴 규격 스켈레톤 선언
        default_result = {"is_malicious": False, "raw_score": 0.0, "detected_count": 0, "status": "safe"}

        if not self.api_key:
            logger.warning("[Google Safe Browsing] API Key가 누락되었습니다. 빈 분석 결과를 반환합니다.")
            return default_result

        payload = {
            "client": {
                "clientId": "safefam-ai-backend",
                "clientVersion": "1.0.0"
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE", 
                    "SOCIAL_ENGINEERING", 
                    "UNWANTED_SOFTWARE", 
                    "POTENTIALLY_HARMFUL_APPLICATION"
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [
                    {"url": url}
                ] 
            }
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_url, json=payload, timeout=5.0)
                response.raise_for_status()

                result = response.json()

                # 응답 데이터에 'matches' 필드가 있으면 확실한 악성 사이트 상태
                if "matches" in result and len(result["matches"]) > 0:
                    logger.warning(f"[Google Safe Browsing] 악성 URL 감지됨: {url}")
                    return {
                        "is_malicious": True,
                        "raw_score": 0.95,  
                        "detected_count": len(result["matches"]),
                        "status": "completed"
                    }
                
                logger.info(f"[Google Safe Browsing] 안전한 URL: {url}")
                return default_result

            except httpx.HTTPStatusError as e:
                logger.error(f"[Google Safe Browsing] API 에러 ({e.response.status_code}): {str(e)}")
                return default_result
            except httpx.TimeoutException:
                logger.error("[Google Safe Browsing] API 요청 타임아웃 발생")
                return default_result
            except Exception as e:
                logger.error(f"[Google Safe Browsing] 연동 중 비정상 에러 발생: {str(e)}")
                return default_result
