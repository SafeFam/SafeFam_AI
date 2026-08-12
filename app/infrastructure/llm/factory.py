"""LLM 공급자 구현체 생성"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.infrastructure.llm.bedrock_client import (
    BedrockLlmClient,
)
from app.infrastructure.llm.types import (
    LlmClient,
)

@lru_cache(maxsize=1)
def get_llm_client() -> LlmClient:
    """설정된 LLM 구현체 반환"""

    if settings.LLM_PROVIDER == "bedrock":
        return BedrockLlmClient()

    raise ValueError(
        f"unsupported LLM provider: "
        f"{settings.LLM_PROVIDER}"
    )

def reset_llm_client_for_test() -> None:
    """테스트 간 singleton client 초기화"""

    get_llm_client.cache_clear()