"""동일한 샘플을 Stacking, LLM 및 Hybrid 모드로 평가하는 실행기"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.analysis.hybrid_policy import ConditionalLlmPolicy
from app.analysis.text.hybrid_analyzer import HybridTextAnalyzer

from .models import EvaluationMode, EvaluationRecord

StackingAnalyzer = Callable[[str], dict[str, Any]]
LlmAnalyzer = Callable[[str], Awaitable[dict[str, Any]]]

_BINARY_LABELS = {"normal", "phishing"}
_PHISHING_LLM_GRADES = {"SUSPICIOUS", "DANGEROUS"}


@dataclass(frozen=True)
class EvaluationSample:
    """평가 실행 중에만 원문을 보유하는 입력 샘플"""

    sample_id: str
    text: str
    expected_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.expected_label not in _BINARY_LABELS:
            raise ValueError("expected_label must be normal or phishing")


class HybridEvaluationRunner:
    """주입된 분석기로 세 가지 평가 모드를 동일하게 실행"""

    def __init__(
        self,
        *,
        policy: ConditionalLlmPolicy,
        stacking_analyzer: StackingAnalyzer,
        llm_analyzer: LlmAnalyzer,
    ) -> None:
        self._stacking_analyzer = stacking_analyzer
        self._llm_analyzer = llm_analyzer
        self._hybrid_analyzer = HybridTextAnalyzer(
            policy=policy,
            stacking_analyzer=stacking_analyzer,
            llm_analyzer=llm_analyzer,
        )

    async def evaluate(
        self,
        samples: Sequence[EvaluationSample],
        *,
        modes: Sequence[EvaluationMode] = tuple(EvaluationMode),
    ) -> list[EvaluationRecord]:
        """샘플 순서와 모드 순서를 보존하며 평가"""
        if not samples:
            raise ValueError("samples must not be empty")
        if not modes:
            raise ValueError("modes must not be empty")
        if len(set(modes)) != len(modes):
            raise ValueError("modes must not contain duplicates")
        if any(not isinstance(mode, EvaluationMode) for mode in modes):
            raise TypeError("modes must contain EvaluationMode values")

        records: list[EvaluationRecord] = []
        for sample in samples:
            if not isinstance(sample, EvaluationSample):
                raise TypeError("samples must contain EvaluationSample values")
            for mode in modes:
                records.append(await self.evaluate_one(sample, mode))
        return records

    async def evaluate_one(
        self,
        sample: EvaluationSample,
        mode: EvaluationMode,
    ) -> EvaluationRecord:
        if not isinstance(sample, EvaluationSample):
            raise TypeError("sample must be an EvaluationSample")
        if not isinstance(mode, EvaluationMode):
            raise TypeError("mode must be an EvaluationMode")

        started_at = perf_counter()
        if mode is EvaluationMode.SELF_MODEL_ONLY:
            outcome = await self._run_self_model(sample.text)
        elif mode is EvaluationMode.LLM_ONLY:
            outcome = await self._run_llm(sample.text)
        else:
            outcome = await self._run_hybrid(sample.text)
        latency_ms = (perf_counter() - started_at) * 1_000
        if outcome.get("llm_from_cache") is True:
            cached_latency_ms = outcome.get("llm_latency_ms")
            if cached_latency_ms is not None:
                latency_ms += cached_latency_ms

        return EvaluationRecord(
            sample_id=sample.sample_id,
            mode=mode,
            expected_label=sample.expected_label,
            predicted_label=outcome["predicted_label"],
            latency_ms=latency_ms,
            result_available=outcome["result_available"],
            llm_called=outcome["llm_called"],
            llm_available=outcome["llm_available"],
            fallback_applied=outcome["fallback_applied"],
            all_engines_unavailable=outcome["all_engines_unavailable"],
            decision_source=outcome["decision_source"],
            routing_decision=outcome.get("routing_decision"),
            routing_reason=outcome.get("routing_reason"),
            error_code=outcome.get("error_code"),
            llm_provider=outcome.get("llm_provider"),
            llm_model=outcome.get("llm_model"),
            input_tokens=outcome.get("input_tokens"),
            output_tokens=outcome.get("output_tokens"),
        )

    async def _run_self_model(self, text: str) -> dict[str, Any]:
        try:
            analysis = await asyncio.to_thread(self._stacking_analyzer, text)
        except Exception:  # 평가 결과에는 예외 문자열을 저장 X
            analysis = {}

        result = analysis.get("result") or {}
        prediction = result.get("is_suspected_phishing")
        available = analysis.get("is_available") is True and isinstance(
            prediction, bool
        )
        error_code = result.get("error_message")
        if not available and not error_code:
            error_code = "STACKING_MODEL_UNAVAILABLE"

        return {
            "predicted_label": (
                "phishing" if prediction else "normal"
            ) if available else "unknown",
            "result_available": available,
            "llm_called": False,
            "llm_available": False,
            "fallback_applied": False,
            "all_engines_unavailable": not available,
            "decision_source": "STACKING" if available else "UNAVAILABLE",
            "error_code": error_code,
        }

    async def _run_llm(self, text: str) -> dict[str, Any]:
        try:
            analysis = await self._llm_analyzer(text)
        except Exception:
            analysis = {}

        result = analysis.get("result") or {}
        predicted_label = _label_from_llm_grade(result.get("grade"))
        available = predicted_label is not None and not result.get("error_message")
        usage = analysis.get("usage") or {}
        error_code = result.get("error_message")
        if not available and not error_code:
            error_code = "LLM_ANALYZER_FAILED"

        return {
            "predicted_label": predicted_label or "unknown",
            "result_available": available,
            "llm_called": True,
            "llm_available": available,
            "fallback_applied": False,
            "all_engines_unavailable": not available,
            "decision_source": "LLM" if available else "UNAVAILABLE",
            "error_code": error_code,
            "llm_provider": analysis.get("provider"),
            "llm_model": analysis.get("model_id"),
            "input_tokens": _optional_token_count(usage.get("input_tokens")),
            "output_tokens": _optional_token_count(usage.get("output_tokens")),
            "llm_latency_ms": _optional_latency(analysis.get("latency_ms")),
            "llm_from_cache": analysis.get("is_cached") is True,
        }

    async def _run_hybrid(self, text: str) -> dict[str, Any]:
        analysis = await self._hybrid_analyzer.analyze(text)
        result = analysis.get("result") or {}
        llm_result = analysis.get("llm") or {}
        predicted_label = _label_from_llm_grade(result.get("grade"))
        available = predicted_label is not None and not result.get("error_message")
        usage = analysis.get("llm_usage") or {}
        all_unavailable = (
            analysis.get("decision_source") == "UNAVAILABLE" or not available
        )

        return {
            "predicted_label": predicted_label or "unknown",
            "result_available": available,
            "llm_called": analysis.get("llm_called") is True,
            "llm_available": analysis.get("llm_available") is True,
            "fallback_applied": analysis.get("fallback_applied") is True,
            "all_engines_unavailable": all_unavailable,
            "decision_source": analysis.get("decision_source", "UNAVAILABLE"),
            "routing_decision": analysis.get("routing_decision"),
            "routing_reason": analysis.get("routing_reason"),
            "error_code": (
                result.get("error_message") or llm_result.get("error_message")
            ),
            "llm_provider": analysis.get("llm_provider"),
            "llm_model": analysis.get("llm_model"),
            "input_tokens": _optional_token_count(usage.get("input_tokens")),
            "output_tokens": _optional_token_count(usage.get("output_tokens")),
            "llm_latency_ms": _optional_latency(
                analysis.get("llm_latency_ms")
            ),
            "llm_from_cache": analysis.get("llm_from_cache") is True,
        }


def _label_from_llm_grade(grade: object) -> str | None:
    if grade == "SAFE":
        return "normal"
    if grade in _PHISHING_LLM_GRADES:
        return "phishing"
    return None


def _optional_token_count(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _optional_latency(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    ):
        return float(value)
    return None
