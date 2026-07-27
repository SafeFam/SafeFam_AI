"""Low-level Gemini HTTP client shared by analysis and chat features."""

from typing import Any

import httpx


class GeminiClient:
    """Send raw generation requests without applying feature-specific policy."""

    async def generate(
        self,
        *,
        api_url: str,
        api_key: str,
        payload: dict[str, Any],
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        headers = {
            "x-goog-api-key": api_key,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
