import asyncio
import time


# Gemini 무료 티어 RPM 한도가 낮아 --concurrency만으로는 429를 못 막는다.
# 실제 호출 간 최소 간격을 강제하는 전역 리미터로 모든 Gemini 호출 지점(변형 생성 + 파이프라인 평가)을 공유시킨다.
class RateLimiter:
    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


GEMINI_RATE_LIMITER = RateLimiter(min_interval_seconds=6.0)
