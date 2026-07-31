import re
import pytest

from app.analysis.text.naive_bayes_analyzer import analyze_text_with_naive_bayes

# Spring PiiMaskingService와 동일한 패턴
_RRN = re.compile(r"(?<!\d)\d{6}[- ]\d{7}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|\d{4}[- ]?\d{6}[- ]?\d{5})(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:0\d{1,2}[- ]?\d{3,4}[- ]?\d{4}|0\d{9,10})(?!\d)")
_ACCOUNT = re.compile(r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,6}(?:-\d{1,6})?(?!\d)|(?<!\d)\d{10,14}(?!\d)")
_EMAIL = re.compile(r"(?i)[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}")
_URL = re.compile(r"(?i)(?<!@)(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+;=%]+")


def mask(text: str) -> str:
    parts = []
    last_end = 0
    for m in _URL.finditer(text):
        parts.append(_mask_pii(text[last_end:m.start()]))
        parts.append(m.group())
        last_end = m.end()
    parts.append(_mask_pii(text[last_end:]))
    return "".join(parts)


def _mask_pii(text: str) -> str:
    text = _RRN.sub("[RRN]", text)
    text = _CARD.sub("[CARD]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _ACCOUNT.sub("[ACCOUNT]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    return text


SAMPLES = [
    "[국민은행] 계좌 110-1234-567890이 정지되었습니다. 즉시 010-1234-5678로 연락하세요.",
    "고객님 명의로 이상 거래가 감지되었습니다. 즉시 확인하세요. http://bit.ly/fake",
    "신한카드 비정상 결제 감지. 1234-5678-9012-3456 카드를 즉시 정지하세요.",
    "[금융감독원] 명의도용 확인 요망. 010-9876-5432로 연락하세요.",
    "계좌 100123456789에서 출금 시도가 감지되었습니다.",
    "대출 승인 완료. 즉시 송금 바랍니다. 110-2345-678901",
    "[검찰청] 귀하의 계좌가 범죄에 연루되었습니다. 즉시 확인하세요.",
    "카드 도용 의심. 즉시 1588-1234로 신고하세요.",
    "오늘 저녁 메뉴 뭐야?",
    "내일 회의 몇 시야?",
]


@pytest.mark.asyncio
async def test_nb_masking_accuracy():
    results = []

    for text in SAMPLES:
        masked = mask(text)
        original_result = await analyze_text_with_naive_bayes(text)
        masked_result = await analyze_text_with_naive_bayes(masked)

        assert original_result["is_available"], "NB 모델 로드 실패 — 원문 추론 불가"
        assert masked_result["is_available"], "NB 모델 로드 실패 — 마스킹 추론 불가"

        original_score = original_result["result"]["risk_score"]
        masked_score = masked_result["result"]["risk_score"]

        diff = abs(original_score - masked_score)

        results.append({
            "original": text,
            "masked": masked,
            "original_score": original_score,
            "masked_score": masked_score,
            "diff": diff,
        })

        print(f"\n원문:     {text[:50]}...")
        print(f"마스킹:   {masked[:50]}...")
        print(f"원문 점수: {original_score} | 마스킹 점수: {masked_score} | 차이: {diff}")

    avg_diff = sum(r["diff"] for r in results) / len(results)
    max_diff = max(r["diff"] for r in results)
    over_threshold = [r for r in results if r["diff"] > 5]

    print(f"\n{'='*60}")
    print(f"평균 점수 차이: {avg_diff:.1f}점")
    print(f"최대 점수 차이: {max_diff}점")
    print(f"5점 초과 케이스: {len(over_threshold)}건 / {len(results)}건")

    if over_threshold:
        print("\n[5점 초과 케이스]")
        for r in over_threshold:
            print(f"  원문: {r['original'][:50]}")
            print(f"  원문 점수: {r['original_score']} | 마스킹 점수: {r['masked_score']} | 차이: {r['diff']}")

    assert avg_diff <= 5, f"NB 정확도 하락이 허용 범위 초과: 평균 {avg_diff:.1f}점 차이"