import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.service import ChatService, ChatServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_chat_service() -> ChatService:
    return ChatService()


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="[멀티턴 챗봇] 분석 결과 기반 금융사기 대응 상담",
)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    # 대화 내용/분석 컨텍스트는 로그에 원문으로 남기지 않음
    logger.info(f"[Router] 챗봇 요청 진입 - 메시지 수: {len(payload.messages)}")

    try:
        return await chat_service.get_response(payload)
    except ChatServiceError as e:
        logger.error(f"[Router] 챗봇 응답 생성 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="챗봇 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )
