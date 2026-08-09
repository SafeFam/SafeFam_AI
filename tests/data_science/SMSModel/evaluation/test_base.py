"""공통 모델 인터페이스와 score 계약 테스트."""

import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.modeling import ScoreOutput, ScoreType


def test_probability_scores_accept_zero_and_one():
    output = ScoreOutput(
        values=np.asarray([0.0, 0.5, 1.0]),
        score_type=ScoreType.PROBABILITY,
    )
    assert output.values.tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_probability_scores_reject_out_of_range_values(invalid_value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ScoreOutput(
            values=np.asarray([invalid_value]),
            score_type=ScoreType.PROBABILITY,
        )


def test_decision_scores_allow_unbounded_values():
    output = ScoreOutput(
        values=np.asarray([-3.5, 0.0, 7.2]),
        score_type=ScoreType.DECISION,
    )
    assert output.values.tolist() == [-3.5, 0.0, 7.2]


@pytest.mark.parametrize(
    "values, message",
    [
        (np.asarray([[0.1, 0.2]]), "one-dimensional"),
        (np.asarray([0.1, np.nan]), "finite"),
        (np.asarray([0.1, np.inf]), "finite"),
    ],
)
def test_score_output_rejects_invalid_shape_or_values(values, message):
    with pytest.raises(ValueError, match=message):
        ScoreOutput(values=values, score_type=ScoreType.PROBABILITY)


def test_predict_uses_probability_default_threshold(fake_model):
    fake_model.fit(pd.DataFrame({"label": ["normal"], "mock_score": [0.1]}))
    frame = pd.DataFrame({"mock_score": [0.49, 0.5, 0.51]})

    assert fake_model.default_threshold == 0.5
    assert fake_model.predict(frame).tolist() == [
        "normal",
        "phishing",
        "phishing",
    ]


def test_predict_accepts_custom_threshold_and_metadata(fake_model):
    fake_model.fit(pd.DataFrame({"label": ["normal"], "mock_score": [0.1]}))
    frame = pd.DataFrame({"mock_score": [0.6, 0.8]})

    assert fake_model.predict(frame, threshold=0.7).tolist() == [
        "normal",
        "phishing",
    ]
    assert fake_model.get_metadata() == {
        "model_name": "fake_probability",
        "score_type": "probability",
        "default_threshold": 0.5,
        "test_adapter": True,
    }
