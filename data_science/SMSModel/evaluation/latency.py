"""모델별 단건 추론 시간 측정"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
import pandas as pd

from data_science.SMSModel.modeling import (
    BasePhishingClassifier,
)


@dataclass(frozen=True)
class LatencyMetrics:
    """밀리초 단위 단건 추론 지연시간 통계"""

    sample_count: int
    warmup_count: int
    average_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


def measure_single_inference_latency(
    model: BasePhishingClassifier,
    df: pd.DataFrame,
    *,
    sample_count: int = 100,
    warmup_count: int = 5,
) -> LatencyMetrics:
    """DataFrame에서 일부 샘플을 선택해 한 건씩 end-to-end 추론 시간 측정"""
    if df.empty:
        raise ValueError("cannot measure latency from an empty DataFrame")

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than 0")

    if warmup_count < 0:
        raise ValueError("warmup_count must not be negative")

    measured_count = min(sample_count, len(df))

    # 항상 같은 샘플을 사용하도록 앞에서부터 선택
    samples = df.iloc[:measured_count]

    # 최초 호출의 lazy initialization과 캐시 영향을 측정에서 제외
    for warmup_index in range(warmup_count):
        sample = samples.iloc[
            warmup_index % measured_count : (warmup_index % measured_count) + 1
        ]
        model.predict_scores(sample)

    durations_ms: list[float] = []

    for row_index in range(measured_count):
        single_row = samples.iloc[row_index : row_index + 1]

        started_at = perf_counter_ns()
        score_output = model.predict_scores(single_row)
        finished_at = perf_counter_ns()

        if len(score_output.values) != 1:
            raise ValueError("single-row inference must return one score")

        durations_ms.append((finished_at - started_at) / 1_000_000)

    duration_array = np.asarray(
        durations_ms,
        dtype=float,
    )

    return LatencyMetrics(
        sample_count=measured_count,
        warmup_count=warmup_count,
        average_ms=float(duration_array.mean()),
        median_ms=float(np.median(duration_array)),
        p95_ms=float(np.percentile(duration_array, 95)),
        minimum_ms=float(duration_array.min()),
        maximum_ms=float(duration_array.max()),
    )
