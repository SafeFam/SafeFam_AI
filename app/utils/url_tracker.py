import re
import logging
from typing import List
from urllib.parse import urljoin
import httpx

logger = logging.getLogger(__name__)

# URL 정규표현식 패턴
URL_PATTERN = re.compile(r'https?://[^\s\'"<>]+')

# 텍스트에서 URL를 추출하고 정제, 중복 제거하여 반환
def extract_urls(text: str) -> List[str]:
    
    if not text:
        return []
    
    raw_urls = URL_PATTERN.findall(text)
    cleaned_urls = []

    for url in raw_urls:
        cleaned_url = url.rstrip('.,?!:;)[]')
        cleaned_urls.append(cleaned_url)

    return list(dict.fromkeys(cleaned_urls))


# 단축 URL의 리다이렉트를 추적하고 최종 주소를 반환
async def trace_url(url: str, max_redirects: int = 5, timeout: float = 3.0) -> str:
   
    current_url = url

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    async with httpx.AsyncClient(limits=limits, follow_redirects=False) as client:
        for attempt in range(max_redirects):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                
                response = await client.head(current_url, headers=headers, timeout=timeout)

                # HEAD를 차단하거나 거부하는 서버(400, 404, 405)에 대응하기 위한 GET 폴백
                if response.status_code in [400, 404, 405]:
                    response = await client.get(current_url, headers=headers, timeout=timeout)

                # HTTP Redirection 상태 코드 판별 (3xx)
                if response.is_redirect or response.status_code in [301, 302, 303, 307, 308]:
                    location = response.headers.get("Location")
                    if not location:
                        break

                    if location.startswith("/"):
                        location = urljoin(current_url, location)

                    current_url = location
                    logger.info(f"Redirect {attempt + 1}: -> {current_url}")
                else:
                    break

            except httpx.TimeoutException:
                logger.warning(f"URL 추적 타임아웃 발생 ({timeout}초 초과): {current_url}")
                break
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP 에러 발생 ({e.response.status_code}): {current_url}")
                break
            except Exception as e:
                logger.error(f"비정상 URL 추적 실패 ({str(e)}): {current_url}")
                break
        else:
            logger.warning(f"최대 리다이렉트 횟수({max_redirects}회)를 초과했습니다. 루프 위험 감지.")

    return current_url