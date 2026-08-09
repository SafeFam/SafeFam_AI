"""Low-level Gemini HTTP client shared by analysis and chat features."""

from typing import Any

import httpx

from app.core.config import settings
from app.infrastructure.http_retry import request_with_retry


class GeminiClient:
    """Send raw generation requests without applying feature-specific policy."""

    async def generate(
        self,
        *,
        api_url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        headers = {
            "x-goog-api-key": api_key,
            "content-type": "application/json",
        }

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.GEMINI_TIMEOUT_SECONDS
        )

        async with httpx.AsyncClient() as client:
            response = await request_with_retry(
                lambda: client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ),
                max_retries=settings.EXTERNAL_API_MAX_RETRIES,
                operation_name="Gemini",
            )

            response.raise_for_status()
            return response.json()
