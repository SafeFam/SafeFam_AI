"""사전학습 인코더의 CPU 추론 지연시간을 채택 기준 예산과 대조"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_science.SMSModel.evaluation.adoption import AdoptionCriteria
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    load_data,
    select_real_holdout,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "encoder_latency_benchmark.json"
)

DEFAULT_MODEL_ID = "beomi/KcELECTRA-small-v2022"
DEFAULT_MAX_LENGTHS = (128, 256)

WARMUP_ITERATIONS = 10

SAMPLE_COUNT = 200

REPORT_SCHEMA_VERSION = 1


def load_messages(sample_count: int) -> list[str]:
    """판정셋 문자를 길이 분포 그대로 가져오기"""
    _, holdout = load_data(DATA_PATH)
    judging = select_real_holdout(holdout)
    texts = judging["text"].astype(str).tolist()

    if len(texts) <= sample_count:
        return texts

    step = len(texts) / sample_count
    return [texts[int(index * step)] for index in range(sample_count)]


def measure_truncation(
    tokenizer,
    messages: list[str],
    max_length: int,
) -> dict[str, float]:
    """max_length에서 잘려나가는 문자의 비율과 정도"""
    lengths = [
        len(tokenizer(text, truncation=False)["input_ids"])
        for text in messages
    ]
    truncated = [length for length in lengths if length > max_length]

    return {
        "token_length_median": float(statistics.median(lengths)),
        "token_length_max": int(max(lengths)),
        "truncated_count": len(truncated),
        "truncated_share": len(truncated) / len(lengths),
        "dropped_token_share": (
            float(np.mean([1 - max_length / length for length in truncated]))
            if truncated
            else 0.0
        ),
    }


def measure_latency(
    model,
    tokenizer,
    messages: list[str],
    max_length: int,
) -> dict[str, float]:
    """단건 추론 지연시간. 운영과 같이 한 건씩 처리"""
    import torch

    def infer(text: str) -> None:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            model(**encoded)

    for text in messages[:WARMUP_ITERATIONS]:
        infer(text)

    durations_ms: list[float] = []
    for text in messages:
        started = time.perf_counter()
        infer(text)
        durations_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "sample_count": len(durations_ms),
        "p50_ms": float(np.percentile(durations_ms, 50)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "p99_ms": float(np.percentile(durations_ms, 99)),
        "mean_ms": float(np.mean(durations_ms)),
        "max_ms": float(max(durations_ms)),
    }


def measure_current_pipeline(model_path: Path | None) -> float | None:
    """현재 stacking artifact가 이미 쓰고 있는 p95. 남은 예산 계산에 필요"""
    if model_path is None or not model_path.is_file():
        return None

    from data_science.SMSModel.evaluation.stacking_reporting import (
        predict_probabilities_with_latency,
    )
    from data_science.SMSModel.run_error_analysis import load_classifier

    _, holdout = load_data(DATA_PATH)
    classifier = load_classifier(model_path)
    _, _, latencies_ms = predict_probabilities_with_latency(
        classifier,
        select_real_holdout(holdout),
    )

    return float(np.percentile(latencies_ms, 95))


def build_report(
    model_id: str,
    max_lengths: tuple[int, ...],
    messages: list[str],
    current_p95_ms: float | None,
) -> dict[str, object]:
    """후보 max_length마다 지연시간과 절단 정도를 측정"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    budget_ms = AdoptionCriteria().max_p95_latency_ms
    remaining_ms = (
        budget_ms - current_p95_ms if current_p95_ms is not None else None
    )

    measurements = []
    for max_length in max_lengths:
        latency = measure_latency(model, tokenizer, messages, max_length)
        truncation = measure_truncation(tokenizer, messages, max_length)
        measurements.append(
            {
                "max_length": max_length,
                "latency": latency,
                "truncation": truncation,
                "fits_remaining_budget": (
                    None
                    if remaining_ms is None
                    else bool(latency["p95_ms"] <= remaining_ms)
                ),
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "model_id": model_id,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "torch_threads": int(torch.get_num_threads()),
        "budget": {
            "max_p95_latency_ms": budget_ms,
            "current_pipeline_p95_ms": current_p95_ms,
            "remaining_ms": remaining_ms,
        },
        "measurements": measurements,
    }


def print_summary(report: dict[str, object]) -> None:
    """관문 통과 여부를 판단할 수 있을 만큼만 출력"""
    budget = report["budget"]
    print(f"[Encoder] {report['model_id']}")
    print(
        f"  파라미터 {report['parameter_count']:,} | "
        f"torch threads {report['torch_threads']}"
    )
    print(
        f"  예산 {budget['max_p95_latency_ms']:.0f}ms"
        f" - 현재 파이프라인 {budget['current_pipeline_p95_ms']:.2f}ms"
        f" = 남은 {budget['remaining_ms']:.2f}ms"
        if budget["current_pipeline_p95_ms"] is not None
        else f"  예산 {budget['max_p95_latency_ms']:.0f}ms (현재 파이프라인 미측정)"
    )

    for measurement in report["measurements"]:
        latency = measurement["latency"]
        truncation = measurement["truncation"]
        verdict = {True: "통과", False: "초과", None: "판정불가"}[
            measurement["fits_remaining_budget"]
        ]
        print(
            f"\n  [max_length={measurement['max_length']}] {verdict}"
            f"\n    p50 {latency['p50_ms']:.2f}ms"
            f" | p95 {latency['p95_ms']:.2f}ms"
            f" | p99 {latency['p99_ms']:.2f}ms"
            f" | max {latency['max_ms']:.2f}ms"
            f"\n    절단 {truncation['truncated_count']}건"
            f" ({truncation['truncated_share'] * 100:.1f}%)"
            f" | 토큰 길이 중앙값 {truncation['token_length_median']:.0f}"
            f" 최대 {truncation['token_length_max']}"
        )


def main() -> None:
    """CLI 인자를 읽어 지연시간 벤치마크를 JSON으로 저장"""
    parser = argparse.ArgumentParser(
        description="Benchmark encoder CPU latency against the adoption budget."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_LENGTHS),
    )
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--stacking-model-path", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args()

    messages = load_messages(arguments.sample_count)
    current_p95_ms = measure_current_pipeline(arguments.stacking_model_path)
    report = build_report(
        arguments.model_id,
        tuple(arguments.max_lengths),
        messages,
        current_p95_ms,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[Encoder] {arguments.output}")
    print_summary(report)


if __name__ == "__main__":
    main()
 