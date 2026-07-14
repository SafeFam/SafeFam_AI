import httpx
import logging

logger = logging.getLogger(__name__)

# 단축 URL의 HTTP 리다이렉트를 추적하여 최종 원본 주소를 반환
async def resolve_short_url(url: str, max_redirects: int = 5, timeout: float = 5.0) -> str:

    cleaned_url = url

    # 비동기 클라이언트 생성
    async with httpx.AsyncClient(fllow_redirects=False) as client:
        for attempt in range(max_redirects):
            try:
                # HEAD 요청을 통한 네트워크 오버헤드 최소화 및 헤더 데이터 수집
                response = await client.head(current_url, timeout=timeout)

                if response.status_code in [400, 404, 405]:
                    response = await client.get(current_url, timeout=timeout)

                # HTTP Redirection 상태 코드 판별 및 Location 헤더 추적
                if response.status_code in [301, 302, 303, 307, 308]:
                    location = response.headers.get("Location")
                    if not location:
                        break

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