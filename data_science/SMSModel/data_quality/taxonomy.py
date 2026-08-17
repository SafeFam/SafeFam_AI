"""SMS 보조 라벨 taxonomy와 legacy type 정규화 규칙"""

from __future__ import annotations

NORMAL_MESSAGE_TYPES = frozenset(
    {
        "일상대화",
        "정상금융알림",
        "정상카드결제알림",
        "정상택배배송안내",
        "정상인증알림",
        "정상포인트소멸알림",
        "정상광고프로모션",
        "정상공공기관알림",
        "기타정상",
    }
)

PHISHING_MESSAGE_TYPES = frozenset(
    {
        "정부공공기관사칭",
        "수사기관사칭",
        "금융기관사칭",
        "대출사기",
        "택배배송사칭",
        "결제환불사칭",
        "지인가족사칭",
        "경조사사칭",
        "이벤트당첨사칭",
        "채용부업사기",
        "투자리딩방사기",
        "중고거래사기",
        "계정정지본인인증유도",
        "악성링크앱설치유도",
        "복합피싱",
        # 분류 근거가 부족한 데이터만 임시로 허용
        # 최종 학습 데이터에서는 가능한 한 제거하거나 재분류
        "기타피싱",
    }
)

ALLOWED_MESSAGE_TYPES = (
    NORMAL_MESSAGE_TYPES
    | PHISHING_MESSAGE_TYPES
)

# 기존 데이터의 의미가 명확한 type만 자동 정규화
# `기타피싱`은 키워드로 자동 분류하지 않고 사람이 검수
LEGACY_TYPE_ALIASES = {
    "정상알림톡": "기타정상",
    "투자리딩방사칭형": "투자리딩방사기",
    "투자_리딩방": "투자리딩방사기",
    "대출_사기": "대출사기",
    "중고거래_사기": "중고거래사기",
    "채용알바사기형": "채용부업사기",
    "수사기관사칭형": "수사기관사칭",
    "지인사칭": "지인가족사칭",
    "택배사칭": "택배배송사칭",
}

def normalize_legacy_message_type(
        message_type: str,
) -> str:
    """명확한 legacy type을 canonical type으로 변환"""

    if not isinstance(message_type, str):
        raise TypeError("message_type must be a string")

    normalized = message_type.strip()

    if not normalized:
        raise ValueError("message_type must not be empty")

    return LEGACY_TYPE_ALIASES.get(
        normalized,
        normalized,
    )

def is_allowed_label_type_pair(
        label: str,
        message_type: str,
) -> bool:
    """이진 label과 보조 type의 조합이 올바른지 확인"""

    normalized_type = normalize_legacy_message_type(
        message_type
    )

    if label == "normal":
        return normalized_type in NORMAL_MESSAGE_TYPES

    if label == "phishing":
        return normalized_type in PHISHING_MESSAGE_TYPES

    return False