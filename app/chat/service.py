import logging

import httpx

from app.chat.prompts import build_system_prompt
from app.chat.schemas import ChatMessage, ChatRequest, ChatResponse, ChatRole
from app.core.config import settings
from app.infrastructure.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MOCK_ENABLED = settings.MOCK_SECURITY_API

MOCK_RESPONSE_MESSAGE = (
    "[시연용 응답] 해당 문자는 위험도가 높게 분석되었습니다. "
    "발신 기관의 공식 채널로 직접 확인하시고, 의심되는 경우 금융감독원(1332) 또는 "
    "경찰청 사이버수사에 신고해 주세요. 비밀번호나 인증번호는 어떤 경우에도 알려주지 마세요."
)

# Gemini generateContent는 role을 user/model로만 받으므로 assistant -> model 변환
_ROLE_TO_GEMINI_ROLE = {
    ChatRole.USER: "user",
    ChatRole.ASSISTANT: "model",
}


class ChatServiceError(Exception):
    """Gemini 호출 실패 등 챗봇 응답 생성을 계속할 수 없는 상황에서 발생"""


def _build_contents(messages: list[ChatMessage]) -> list[dict]:
    return [
        {
            "role": _ROLE_TO_GEMINI_ROLE[message.role],
            "parts": [{"text": message.content}],
        }
        for message in messages
    ]


class ChatService:
    """분석 컨텍스트 + 대화 히스토리를 Gemini에 전달해 상담 응답을 생성 (Stateless)"""

    async def get_response(self, request: ChatRequest) -> ChatResponse:
        if MOCK_ENABLED:
            logger.info("[Mock Gemini Chat] 실제 API 호출 우회 (Sandbox Mode)")
            return ChatResponse(message=MOCK_RESPONSE_MESSAGE)

        if not GEMINI_API_KEY:
            logger.warning("Gemini API Key가 누락되었습니다.")
            raise ChatServiceError("Missing API Key")

        payload = {
            "system_instruction": {
                "parts": [{"text": build_system_prompt(request.analysisContext)}]
            },
            "contents": _build_contents(request.messages),
        }

        try:
            result_json = await GeminiClient().generate(
                api_url=API_URL,
                api_key=GEMINI_API_KEY,
                payload=payload,
            )

            candidates = result_json.get("candidates", [])
            if not candidates:
                logger.error("[Gemini Chat] 응답에 candidates가 없습니다.")
                raise ChatServiceError("Empty Response")

            message_text = candidates[0]["content"]["parts"][0]["text"]
            logger.info("[Gemini Chat] 응답 생성 완료")
            return ChatResponse(message=message_text)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.error("[Gemini Chat] API 호출 한도 초과 (Rate Limit)")
                raise ChatServiceError("Rate Limit") from e
            logger.error(f"Gemini Chat API 에러 ({e.response.status_code})")
            raise ChatServiceError("HTTP Error") from e

        except httpx.TimeoutException as e:
            logger.error("Gemini Chat API 요청 타임아웃 발생")
            raise ChatServiceError("Timeout") from e

        except (KeyError, IndexError) as e:
            logger.error(f"Gemini Chat 응답 파싱 실패: {type(e).__name__}")
            raise ChatServiceError("Parse Error") from e
