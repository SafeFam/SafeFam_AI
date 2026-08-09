"""Kiwi를 이용한 한국어 SMS 형태소 tokenizer.

이 모듈의 ``kiwi_tokenize`` 함수는 scikit-learn의
CountVectorizer 또는 TfidfVectorizer에서 바로 사용할 수 있도록
문자열을 입력받아 문자열 토큰 리스트를 반환합니다.
"""

from __future__ import annotations

import re
from functools import lru_cache

from kiwipiepy import Kiwi

# 기존 공통 전처리에서 만들어지는 마스킹 토큰입니다.
#
# 형태소 분석기에 "[URL]" 전체를 전달하면 "[", "URL", "]"로
# 분리될 수 있으므로 tokenizer에서 먼저 찾아 하나의 토큰으로 보존합니다.
MASK_TOKENS: frozenset[str] = frozenset(
    {
        "[URL]",
        "[PHONE]",
        "[ACCOUNT]",
        "[AMOUNT]",
        "[EMAIL]",
        "[CARD]",
        "[RRN]",
    }
)

# 정규식에서 긴 토큰이 짧은 토큰보다 먼저 매칭되도록 정렬합니다.
MASK_TOKEN_PATTERN = re.compile(
    "("
    + "|".join(
        re.escape(token)
        for token in sorted(
            MASK_TOKENS,
            key=len,
            reverse=True,
        )
    )
    + ")"
)


# 분류에 사용할 품사를 명시적으로 정의합니다.
#
# NNG: 일반 명사
# NNP: 고유 명사
# NNB: 의존 명사
# NR: 수사
# NP: 대명사
# VV: 동사
# VA: 형용사
# VX: 보조 용언
# VCP: 긍정 지정사
# VCN: 부정 지정사
# MM: 관형사
# MAG: 일반 부사
# MAJ: 접속 부사
# XR: 어근
# SL: 영문/외국어
#
# 조사(J*)와 어미(E*)는 메시지의 핵심 의미보다 문법적인 역할이
# 크기 때문에 기본적으로 제외합니다.
SELECTED_POS_TAGS: frozenset[str] = frozenset(
    {
        "NNG",  # 일반 명사
        "NNP",  # 고유 명사
        "NNB",  # 의존 명사
        "NR",  # 수사
        "NP",  # 대명사
        "VV",  # 동사
        "VA",  # 형용사
        "VX",  # 보조 용언
        "VCP",  # 긍정 지정사
        "VCN",  # 부정 지정사
        "MM",  # 관형사
        "MAG",  # 일반 부사
        "MAJ",  # 접속 부사
        "XR",  # 어근
        "SL",  # 영문/외국어
    }
)


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """현재 Python 프로세스에서 Kiwi 인스턴스를 한 번만 생성"""

    return Kiwi()


def _base_pos_tag(tag: str) -> str:
    """Kiwi의 불규칙 활용 접미사를 제거한 기본 품사를 반환"""

    return tag.split("-", maxsplit=1)[0]


def _tokenize_segment(segment: str) -> list[str]:
    """마스킹 토큰이 포함되지 않은 일반 문자열 조각 분석"""
    if not segment or segment.isspace():
        return []

    tokens: list[str] = []

    # 초성체가 앞 음절의 받침으로 붙어 형태소 분석이 깨지는 경우 최소화 (ex: "했엌ㅋㅋ")
    for token in _get_kiwi().tokenize(
        segment,
        normalize_coda=True,
    ):
        base_tag = _base_pos_tag(token.tag)
        form = token.form.strip()

        if not form:
            continue

        if base_tag not in SELECTED_POS_TAGS:
            continue

        tokens.append(form)

    return tokens


def kiwi_tokenize(text: str) -> list[str]:
    """문자열을 분류용 형태소 토큰 리스트로 변환"""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return []

    result: list[str] = []

    # re.split의 캡처 그룹으로 인해 마스킹 토큰도 결과에 포함
    for part in MASK_TOKEN_PATTERN.split(text):
        if not part or part.isspace():
            continue

        if part in MASK_TOKENS:
            result.append(part)
            continue

        result.extend(_tokenize_segment(part))

    return result
