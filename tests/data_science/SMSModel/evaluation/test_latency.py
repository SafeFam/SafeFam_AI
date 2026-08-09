"""단건 추론 지연시간 측정 테스트."""

import pandas as pd
import pytest

from data_science.SMSModel.evaluation.latency import (
    measure_single_inference_latency,
)


def test_measures_requested_samples_and_warmups(fake_model):
    frame = pd.DataFrame(
        {
            "label": ["normal", "phishing", "normal"],
            "mock_score": [0.1, 0.9, 0.2],
        }
    )
    fake_model.fit(frame)

    metrics = measure_single_inference_latency(
        fake_model,
        frame,
        sample_count=2,
        warmup_count=3,
    )

    assert metrics.sample_count == 2
    assert metrics.warmup_count == 3
    assert fake_model.predict_call_count == 5
    assert metrics.minimum_ms >= 0.0
    assert metrics.minimum_ms <= metrics.average_ms <= metrics.maximum_ms
    assert metrics.p95_ms >= metrics.median_ms


def test_sample_count_is_capped_by_dataframe_size(fake_model):
    frame = pd.DataFrame({"mock_score": [0.2, 0.8]})
    fake_model.fit(frame)
    result = measure_single_inference_latency(
        fake_model,
        frame,
        sample_count=100,
        warmup_count=0,
    )
    assert result.sample_count == 2


@pytest.mark.parametrize(
    "frame, sample_count, warmup_count, message",
    [
        (pd.DataFrame(), 1, 0, "empty"),
        (pd.DataFrame({"mock_score": [0.1]}), 0, 0, "greater than 0"),
        (pd.DataFrame({"mock_score": [0.1]}), 1, -1, "negative"),
    ],
)
def test_rejects_invalid_latency_arguments(
    fake_model,
    frame,
    sample_count,
    warmup_count,
    message,
):
    fake_model.fit(pd.DataFrame({"mock_score": [0.1]}))
    with pytest.raises(ValueError, match=message):
        measure_single_inference_latency(
            fake_model,
            frame,
            sample_count=sample_count,
            warmup_count=warmup_count,
        )
