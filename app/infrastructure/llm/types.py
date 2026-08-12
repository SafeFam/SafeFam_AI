"""외부 LLM 공급자의 공통 입출력 계약"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class LlmGeneration:
    """공급자 응답을 애플리케이션 공통 형태로 정규화"""

    text: str
    provider: str
    model_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None

class LlmClient(Protocal):
    """분석 및 채팅 서비스가 의존할 LLM 인터페이스"""
    async def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LlmGeneration:
        """외부 LLM 응답 생성"""