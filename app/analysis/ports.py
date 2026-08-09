"""Protocols that isolate analysis orchestration from external providers."""

from typing import Any, Protocol


class UrlSecurityProvider(Protocol):
    """Contract implemented by URL reputation providers such as GSB and VT."""

    async def scan_url(self, url: str) -> dict[str, Any]: ...
