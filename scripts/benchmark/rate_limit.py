import asyncio
import time


# 모델별로 RPM 한도가 다르므로(예: gemini-flash-latest는 미확인, gemini-3.1-flash-lite는
# 15RPM) 벤치마크 스크립트 전용으로 최소 호출 간격을 강제하는 간단한 리미터.
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
