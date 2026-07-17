import os 
import logging
import httpx
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
API_URL = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}"

# Google Safe Browsing API를 사용하여 URL의 악성 여부를 1차 검사
async def check_google_safe_browsing(url: str) -> bool:

    if not GOOGLE_API_KEY:
        logger.warning("Google Safe Browsing API Key가 누락되었습니다.")
        return False

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

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, json=payload, timeout=5.0)

            # API 호출 결과 에러 핸들링
            response.raise_for_status()

            result = response.json()

            # 응답 데이터에 'matches' 필드가 있으면 악성 사이트로 등록된 상태
            if "matches" in result and len(result["matches"]) > 0:
                logger.warning(f"[Google Safe Browsing] 악성 URL 감지됨: {url}")
                return True
            
            logger.info(f"[Google Safe Browsing] 안전한 URL: {url}")
            return False

    except httpx.HTTPStatusError as e:
        logger.error(f"Google Safe Browsing API 에러 ({e.response.status_code}): {str(e)}")
        return False

    except httpx.TimeoutException:
        logger.error("Google Safe Browsing API 요청 타임아웃 발생")
        return False
        
    except Exception as e:
        logger.error(f"Google Safe Browsing 연동 중 비정상 에러 발생: {str(e)}")
        return False