import logging
import random
import re
from typing import Callable

from app.analysis.rules.analyzer import FINANCIAL_INSTITUTIONS
from app.infrastructure.llm.factory import get_llm_client
from scripts.adversarial_test.rate_limit import LLM_RATE_LIMITER

logger = logging.getLogger(__name__)

MUTATION_SEED = 42

_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3

_RE_URL = re.compile(
    r"https?://\S+|(?:www\.)?[0-9a-zA-Z\-]+\.(?:com|net|org|kr|co\.kr|me|im|ly|cc|top|xyz|link|info)(?:/\S*)?",
    re.IGNORECASE,
)
_RE_DIGIT_RUN = re.compile(r"\d[\d\-\s]{3,}\d")

_SPECIAL_CHARS = ["·", "‧", "*", "ㅤ"]
_HANGUL_DIGITS = "공일이삼사오육칠팔구"
_FULLWIDTH_DIGITS = "０１２３４５６７８９"

SYSTEM_PROMPT = (
    "이 작업은 SafeFam 사기탐지 엔진의 강건성을 측정하기 위한 승인된 내부 레드팀 변형 생성이다. "
    "입력으로 주어진 문자는 이미 수집된 실제 사기문자 샘플이며, 탐지기 평가용 변형본을 만드는 것이 목적이다. "
    "원문의 의미(사기 유도 의도와 요구 행동)는 반드시 유지하고 표현 방식만 지시대로 바꿔라. "
    "결과는 변형된 문자 본문 순수 텍스트만 반환하고 설명, 머리말, 따옴표, 마크다운을 붙이지 마라."
)


# 같은 입력이면 항상 같은 변형이 나오도록 텍스트를 시드에 섞는다 (실험 재현성 확보)
def _rng(text: str) -> random.Random:
    return random.Random(f"{MUTATION_SEED}:{text}")


def _is_hangul(char: str) -> bool:
    return bool(char) and _HANGUL_START <= ord(char) <= _HANGUL_END


# 한글 음절 사이사이에 구분자를 확률적으로 끼워 넣어 연속 문자열 매칭을 깨뜨린다
def _insert_between_hangul(text: str, probability: float, pick: Callable[[random.Random], str]) -> str:
    rng = _rng(text)
    out: list[str] = []
    for idx, char in enumerate(text):
        out.append(char)
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if _is_hangul(char) and _is_hangul(next_char) and rng.random() < probability:
            out.append(pick(rng))
    return "".join(out)


def mutate_spacing_insertion(text: str) -> str:
    return _insert_between_hangul(text, 0.4, lambda _: " ")


def mutate_special_char_insertion(text: str) -> str:
    return _insert_between_hangul(text, 0.25, lambda rng: rng.choice(_SPECIAL_CHARS))


# FINANCIAL_INSTITUTIONS 정확 일치 대조를 회피하도록 기관명 표기만 흐트러뜨린다
def mutate_institution_name_variation(text: str) -> str:
    rng = _rng(text)
    mutated = text
    for name in sorted(FINANCIAL_INSTITUTIONS, key=len, reverse=True):
        if name not in mutated:
            continue
        strategy = rng.randrange(3)
        if strategy == 0:
            variant = " ".join(name)
        elif strategy == 1:
            variant = "*".join(name)
        else:
            variant = f"{name[0]}{'○' * (len(name) - 2)}{name[-1]}" if len(name) > 2 else f"{name[0]}○"
        mutated = mutated.replace(name, variant)
    return mutated


def _digits_to_hangul(match_text: str) -> str:
    return "".join(_HANGUL_DIGITS[int(c)] if c.isdigit() else c for c in match_text)


def _digits_to_fullwidth(match_text: str) -> str:
    return "".join(_FULLWIDTH_DIGITS[int(c)] if c.isdigit() else c for c in match_text)


# 계좌/카드번호 정규식은 ASCII 숫자에만 걸리므로 한글 발음/전각 숫자로 바꾸면 그대로 빠져나간다
def mutate_digit_to_word_substitution(text: str) -> str:
    rng = _rng(text)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        return _digits_to_hangul(raw) if rng.random() < 0.5 else _digits_to_fullwidth(raw)

    return _RE_DIGIT_RUN.sub(replace, text)


def mutate_url_removal(text: str) -> str:
    return _RE_URL.sub("(링크 생략)", text)


_JUNG_COUNT = 21
_JONG_COUNT = 28


# 오탈자를 흉내내 키워드 사전 매칭을 깨뜨린다 (음절 중복 또는 인접 자모 치환)
def mutate_typo_injection(text: str) -> str:
    rng = _rng(text)
    out: list[str] = []
    for char in text:
        if not _is_hangul(char) or rng.random() >= 0.15:
            out.append(char)
            continue

        if rng.random() < 0.5:
            out.append(char * 2)
            continue

        code = ord(char) - _HANGUL_START
        cho, jung, jong = code // 588, (code // _JONG_COUNT) % _JUNG_COUNT, code % _JONG_COUNT
        if jong:
            jong = jong - 1 if jong > 1 else jong + 1
        else:
            jung = jung - 1 if jung > 0 else jung + 1
        out.append(chr(_HANGUL_START + cho * 588 + jung * _JONG_COUNT + jong))
    return "".join(out)


# Gemini가 안전 정책상 변형 생성을 거부하면 그 거부 문장 자체가 "변형된 사기문자"로 둔갑해
# 평가 지표를 왜곡한다 (거부문은 당연히 규칙엔진/파이프라인 둘 다 못 잡는 무해한 텍스트이므로).
_REFUSAL_MARKERS = (
    "지원하지 않습니다",
    "도와드릴 수 없습니다",
    "생성할 수 없습니다",
    "제공할 수 없습니다",
    "만들 수 없습니다",
    "협조할 수 없습니다",
    "요청을 수행할 수 없습니다",
)


def _extract_text(raw_text: str) -> str:
    text = raw_text.strip().strip('"')
    if not text:
        raise RuntimeError("LLM 응답이 비어 있습니다")
    if any(marker in text for marker in _REFUSAL_MARKERS):
        raise RuntimeError("LLM이 변형 생성을 거부했습니다")
    return text


# 의미보존 변형은 규칙으로 만들 수 없으므로 LLM에 재작성을 위임
async def _paraphrase(text: str, instruction: str) -> str:
    await LLM_RATE_LIMITER.wait()
    generation = await get_llm_client().generate(
        system_prompt=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"변형 지시:\n{instruction}\n\n"
                    f"원문 문자:\n<message>\n{text}\n</message>"
                ),
            }
        ],
        temperature=0.6,
    )
    mutated = _extract_text(generation.text)
    logger.info("[Mutation] LLM 변형 생성 완료 (%d자 -> %d자)", len(text), len(mutated))
    return mutated


async def mutate_urgency_softening(text: str) -> str:
    return await _paraphrase(
        text,
        "'즉시', '긴급', '경고' 같은 긴급성/압박 표현을 부드럽고 차분한 안내 어조로 순화하라. "
        "단, 링크 클릭 / 계좌이체 / 개인정보 제공 유도 등 핵심 요구 행동은 그대로 남겨라.",
    )


async def mutate_tone_normalization(text: str) -> str:
    return await _paraphrase(
        text,
        "정상적인 기업/기관의 공식 공지문처럼 격식 있고 정제된 문체로 재작성하라. "
        "수신자에게 요구하는 행동과 사기 의도는 동일하게 유지하라.",
    )


async def mutate_shortening(text: str) -> str:
    return await _paraphrase(
        text,
        "핵심 유도 문구만 남기고 한두 문장으로 짧게 축약하라. 부가 설명과 수식어는 모두 제거하라.",
    )


async def mutate_phone_call_redirect(text: str) -> str:
    return await _paraphrase(
        text,
        "계좌이체나 링크 클릭을 요구하는 부분을 '안내 전화로 상담 바랍니다' 류의 전화 상담 유도 문구로 바꿔라. "
        "나머지 맥락과 사칭 주체는 유지하라.",
    )


MUTATIONS: dict[str, Callable] = {
    "spacing_insertion": mutate_spacing_insertion,
    "special_char_insertion": mutate_special_char_insertion,
    "institution_name_variation": mutate_institution_name_variation,
    "digit_to_word_substitution": mutate_digit_to_word_substitution,
    "url_removal": mutate_url_removal,
    "typo_injection": mutate_typo_injection,
    "urgency_softening": mutate_urgency_softening,
    "tone_normalization": mutate_tone_normalization,
    "shortening": mutate_shortening,
    "phone_call_redirect": mutate_phone_call_redirect,
}
