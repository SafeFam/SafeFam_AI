"""형태소 TF-IDF Logistic Regression 분류기 테스트"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.modeling.base import ScoreType
from data_science.SMSModel.modeling.logistic_regression import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_DF,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MAX_ITER,
    DEFAULT_MIN_DF,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_RANDOM_STATE,
    LogisticRegressionPhishingClassifier,
)


@pytest.fixture
def logistic_training_dataframe() -> pd.DataFrame:
    """형태소·bigram이 min_df 조건을 만족하는 균형 학습 데이터."""
    rows: list[dict[str, str]] = []

    for index in range(12):
        rows.append(
            {
                "text": f"오늘 가족 모임 시간을 안내합니다 {index}",
                "label": "normal",
            }
        )
        rows.append(
            {
                "text": (
                    "계좌가 정지되었습니다 즉시 본인 인증하세요 "
                    f"https://bit.ly/fake{index}"
                ),
                "label": "phishing",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def fitted_logistic_model(
    logistic_training_dataframe: pd.DataFrame,
) -> LogisticRegressionPhishingClassifier:
    """여러 테스트에서 재사용할 학습 완료 모델."""
    return LogisticRegressionPhishingClassifier().fit(
        logistic_training_dataframe
    )


def test_fit_and_predict_probability_scores(
    fitted_logistic_model: LogisticRegressionPhishingClassifier,
) -> None:
    """validation/test 형태의 미학습 입력에도 확률 예측이 가능해야 합니다."""
    evaluation_df = pd.DataFrame(
        {
            "text": [
                "내일 병원 예약 시간은 오전 열 시입니다",
                "계좌가 정지되었습니다 즉시 https://danger.example 확인",
            ]
        }
    )

    scores = fitted_logistic_model.predict_scores(evaluation_df)
    predictions = fitted_logistic_model.predict(evaluation_df)

    assert scores.score_type == ScoreType.PROBABILITY
    assert scores.values.shape == (2,)
    assert np.all(scores.values >= 0.0)
    assert np.all(scores.values <= 1.0)
    assert set(predictions).issubset({"normal", "phishing"})


def test_fit_uses_normalize_text_output(
    logistic_training_dataframe: pd.DataFrame,
) -> None:
    """원문의 URL과 개인정보가 공통 전처리 결과로 vectorizer에 전달되어야 합니다."""
    model = LogisticRegressionPhishingClassifier()
    model.fit(logistic_training_dataframe)

    normalized = model._prepare_normalized_text(
        pd.DataFrame(
            {
                "text": [
                    "확인 https://example.com 010-1234-5678 10,000원",
                ]
            }
        )
    )

    assert normalized.iloc[0] == normalize_text(
        "확인 https://example.com 010-1234-5678 10,000원"
    )
    assert "[URL]" in normalized.iloc[0]
    assert "[PHONE]" in normalized.iloc[0]
    assert "[AMOUNT]" in normalized.iloc[0]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "!!! @@@ ###",
        "[URL] [PHONE] [ACCOUNT] [AMOUNT] [EMAIL] [CARD] [RRN]",
    ],
)
def test_predict_handles_empty_special_and_masked_text(
    fitted_logistic_model: LogisticRegressionPhishingClassifier,
    text: str,
) -> None:
    """토큰이 없거나 마스킹 토큰만 있는 입력도 실패하지 않아야 합니다."""
    scores = fitted_logistic_model.predict_scores(pd.DataFrame({"text": [text]}))

    assert scores.values.shape == (1,)
    assert np.isfinite(scores.values[0])
    assert 0.0 <= scores.values[0] <= 1.0


def test_predict_before_fit_fails(
    logistic_training_dataframe: pd.DataFrame,
) -> None:
    """fit 호출 전에는 vectorizer와 classifier를 사용할 수 없어야 합니다."""
    model = LogisticRegressionPhishingClassifier()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_scores(logistic_training_dataframe)


@pytest.mark.parametrize(
    ("dataframe", "error_type", "message"),
    [
        (pd.DataFrame({"label": ["normal"]}), ValueError, "missing"),
        (pd.DataFrame({"text": []}), ValueError, "empty"),
        (pd.DataFrame({"text": [None]}), ValueError, "missing values"),
        (pd.DataFrame({"text": [123]}), TypeError, "must be strings"),
    ],
)
def test_predict_validates_input_dataframe(
    fitted_logistic_model: LogisticRegressionPhishingClassifier,
    dataframe: pd.DataFrame,
    error_type: type[Exception],
    message: str,
) -> None:
    """잘못된 추론 입력은 명확한 예외로 거부해야 합니다."""
    with pytest.raises(error_type, match=message):
        fitted_logistic_model.predict_scores(dataframe)


def test_fit_validates_training_labels() -> None:
    """학습에는 정상과 피싱 클래스가 모두 필요합니다."""
    model = LogisticRegressionPhishingClassifier()

    with pytest.raises(ValueError, match="both normal and phishing"):
        model.fit(
            pd.DataFrame(
                {
                    "text": ["일반 안내", "가족 모임"],
                    "label": ["normal", "normal"],
                }
            )
        )


def test_joblib_round_trip_preserves_scores(
    fitted_logistic_model: LogisticRegressionPhishingClassifier,
    tmp_path: Path,
) -> None:
    """저장 후 로드한 모델은 같은 입력에 동일한 확률을 반환해야 합니다."""
    evaluation_df = pd.DataFrame(
        {
            "text": [
                "오늘 저녁 식사 약속 안내",
                "즉시 계좌 인증 https://danger.example",
                "[국외발신] [ACCOUNT] 입금 요청",
            ]
        }
    )
    expected_scores = fitted_logistic_model.predict_scores(evaluation_df).values
    artifact_path = tmp_path / "logistic_regression.joblib"

    joblib.dump(fitted_logistic_model, artifact_path)
    restored_model = joblib.load(artifact_path)
    restored_scores = restored_model.predict_scores(evaluation_df).values

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=0.0, atol=0.0)


def test_metadata_records_reproducible_configuration(
    fitted_logistic_model: LogisticRegressionPhishingClassifier,
) -> None:
    """보고서와 artifact에 전처리·TF-IDF·분류기 설정이 기록되어야 합니다."""
    metadata = fitted_logistic_model.get_metadata()

    assert metadata["model_name"] == "logistic_regression_morph_tfidf"
    assert metadata["score_type"] == "probability"
    assert metadata["preprocessing"]["normalizer"].endswith("normalize_text")
    assert metadata["preprocessing"]["tokenizer"].endswith("kiwi_tokenize")
    assert metadata["vectorizer"] == {
        "type": "TfidfVectorizer",
        "ngram_range": list(DEFAULT_NGRAM_RANGE),
        "min_df": DEFAULT_MIN_DF,
        "max_df": DEFAULT_MAX_DF,
        "max_features": DEFAULT_MAX_FEATURES,
        "lowercase": False,
        "sublinear_tf": True,
        "token_pattern": None,
    }
    assert metadata["classifier"]["class_weight"] == DEFAULT_CLASS_WEIGHT
    assert metadata["classifier"]["random_state"] == DEFAULT_RANDOM_STATE
    assert metadata["classifier"]["max_iter"] == DEFAULT_MAX_ITER