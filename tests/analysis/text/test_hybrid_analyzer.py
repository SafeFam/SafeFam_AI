"""Stacking + Gemini 하이브리드 분석기 테스트"""

import pytest

from app.analysis.hybrid_policy import (
    ConditionalGeminiPolicy,
    HybridThresholds,
)
from app.analysis.text.hybrid_analyzer import (
    HybridTextAnalyzer,
)


def build_stacking_result(
    probability: float,
) -> dict:
    return {
        "engine": "stacking",
        "is_available": True,
        "result": {
            "risk_probability": probability,
            "risk_score": round(probability * 100),
            "confidence": 0.8,
            "is_suspected_phishing": (
                probability >= 0.5
            ),
            "model_scores": {},
            "unavailable_models": [],
            "error_message": None,
        },
    }


def build_gemini_result(
    score: int = 80,
) -> dict:
    return {
        "is_mock": False,
        "result": {
            "grade": "DANGEROUS",
            "risk_score": score,
            "tone_analysis": "긴급성 유도",
            "evidence": ["긴급성 표현"],
            "reason": "피싱 가능성이 높습니다.",
        },
    }


def build_policy() -> ConditionalGeminiPolicy:
    return ConditionalGeminiPolicy(
        HybridThresholds(
            normal_max=0.2,
            phishing_min=0.8,
        )
    )


@pytest.mark.asyncio
async def test_skips_gemini_for_confident_normal():
    gemini_calls = 0

    async def gemini_analyzer(_text: str):
        nonlocal gemini_calls
        gemini_calls += 1
        return build_gemini_result()

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=(
            lambda _text: build_stacking_result(0.1)
        ),
        gemini_analyzer=gemini_analyzer,
    )

    result = await analyzer.analyze(
        "오늘 저녁 같이 먹자"
    )

    assert gemini_calls == 0
    assert result["gemini_called"] is False
    assert result["decision_source"] == "STACKING"
    assert result["result"]["risk_score"] == 10


@pytest.mark.asyncio
async def test_skips_gemini_for_confident_phishing():
    gemini_calls = 0

    async def gemini_analyzer(_text: str):
        nonlocal gemini_calls
        gemini_calls += 1
        return build_gemini_result()

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=(
            lambda _text: build_stacking_result(0.9)
        ),
        gemini_analyzer=gemini_analyzer,
    )

    result = await analyzer.analyze(
        "즉시 계좌로 송금하세요"
    )

    assert gemini_calls == 0
    assert result["gemini_called"] is False
    assert result["decision_source"] == "STACKING"
    assert result["result"]["risk_score"] == 90


@pytest.mark.asyncio
async def test_calls_gemini_for_uncertain_prediction():
    gemini_calls = 0

    async def gemini_analyzer(_text: str):
        nonlocal gemini_calls
        gemini_calls += 1
        return build_gemini_result(score=85)

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=(
            lambda _text: build_stacking_result(0.5)
        ),
        gemini_analyzer=gemini_analyzer,
    )

    result = await analyzer.analyze(
        "본인 확인이 필요합니다"
    )

    assert gemini_calls == 1
    assert result["gemini_called"] is True
    assert result["gemini_available"] is True
    assert result["decision_source"] == "GEMINI"
    assert result["result"]["risk_score"] == 85


@pytest.mark.asyncio
async def test_uses_stacking_when_gemini_fails():
    async def failed_gemini(_text: str):
        return {
            "result": {
                "grade": "UNKNOWN",
                "risk_score": 0,
                "error_message": "Timeout",
            }
        }

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=(
            lambda _text: build_stacking_result(0.5)
        ),
        gemini_analyzer=failed_gemini,
    )

    result = await analyzer.analyze(
        "본인 확인이 필요합니다"
    )

    assert result["decision_source"] == (
        "STACKING_FALLBACK"
    )
    assert result["fallback_applied"] is True
    assert result["result"]["risk_score"] == 50


@pytest.mark.asyncio
async def test_does_not_fail_open_when_all_engines_fail():
    async def failed_gemini(_text: str):
        return {
            "result": {
                "grade": "UNKNOWN",
                "risk_score": 0,
                "error_message": "Timeout",
            }
        }

    analyzer = HybridTextAnalyzer(
        policy=build_policy(),
        stacking_analyzer=lambda _text: {
            "engine": "stacking",
            "is_available": False,
            "result": {
                "risk_probability": None,
                "risk_score": None,
                "confidence": 0.0,
            },
        },
        gemini_analyzer=failed_gemini,
    )

    result = await analyzer.analyze(
        "테스트 메시지"
    )

    assert result["result"]["grade"] == "UNKNOWN"
    assert result["result"]["risk_score"] is None
    assert (
        result["result"]["error_message"]
        == "ALL_TEXT_ENGINES_UNAVAILABLE"
    )