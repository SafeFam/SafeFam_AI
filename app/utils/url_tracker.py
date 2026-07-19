import re
import logging
from typing import List
from urllib.parse import urljoin
import httpx

# 로깅 설정 (추적 과정 모니터링용)
logger = logging.getLogger(__name__)

# 일반적인 URL 형태를 매칭하기 위한 정규식 패턴
URL_PATTERN = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)

# 문자 텍스트 본문에서 모든 URL을 추출하여 리스트로 반환
def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    
    return list(set(URL_PATTERN.findall(text)))

# 단축 URL의 리다이렉트를 추적하고 최종 도달 주소를 반환
async def trace_url(url: str, max_redirects: int = 5) -> str:
    current_url = url

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    # 비동기 HTTP 클라이언트 설정
    async with httpx.AsyncClient(limits=limits, timeout=3.0, follow_redirects=False) as client:
        for attempt in range(max_redirects):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                response = await client.head(current_url, headers=headers)
                
                if response.is_redirect:
                    next_url = response.headers.get("Location")
                    if not next_url:
                        break
                        
                    if next_url.startswith("/"):
                        next_url = urljoin(current_url, next_url)
                        
                    current_url = next_url
                    logger.info(f"[Redirect {attempt + 1}] -> {current_url}")
                else:
                    break

            except httpx.HTTPStatusError as e:
                if e.response.status_code in [404, 405]:
                    try:
                        response = await client.get(current_url, headers=headers, timeout=2.0)
                        if response.is_redirect:
                            current_url = response.headers.get("Location", current_url)
                    except Exception:
                        break
                break
            except (httpx.HTTPError, Exception) as e:
                logger.error(f"URL 추적 중 예러 발생 ({current_url}): {str(e)}")
                break
    
    return current_url
