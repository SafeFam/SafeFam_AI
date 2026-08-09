"""문자 n-gram TF-IDF Linear SVM 분류기 테스트."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from data_science.SMSModel.evaluation.threshold import (
    select_validation_threshold,
)
from data_science.SMSModel.modeling.base import ScoreType
from data_science.SMSModel.modeling.linear_svm import (
    DEFAULT_C,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_DF,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MAX_ITER,
    DEFAULT_MIN_DF,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_RANDOM_STATE,
    LinearSvmPhishingClassifier,
)


@pytest.fixture
def svm_training_dataframe() -> pd.DataFrame:
    """문자 n-gram이 min_df 조건을 만족하는 균형 학습 데이터."""
    rows: list[dict[str, str]] = []

    for index in range(20):
        rows.append(
            {
                "text": f"오늘 가족 모임 시간을 안내합니다 {index}",
                "text_norm": f"오늘 가족 모임 시간을 안내합니다 {index}",
                "label": "normal",
            }
        )

        rows.append(
            {
                "text": (
                    "계좌가 정지되었습니다 즉시 본인 인증하세요 "
                    f"https://bit.ly/fake{index}"
                ),
                "text_norm": (
                    "계좌가 정지되었습니다 즉시 본인 인증하세요 [URL]"
                ),
                "label": "phishing",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def svm_validation_dataframe() -> pd.DataFrame:
    """threshold 선택에 사용할 별도 validation 데이터."""
    return pd.DataFrame(
        [
            {
                "text": "내일 오전 병원 예약 안내입니다",
                "text_norm": "내일 오전 병원 예약 안내입니다",
                "label": "normal",
            },
            {
                "text": "이번 주 가족 식사 장소 안내",
                "text_norm": "이번 주 가족 식사 장소 안내",
                "label": "normal",
            },
            {
                "text": "택배가 오후 세 시에 도착합니다",
                "text_norm": "택배가 오후 세 시에 도착합니다",
                "label": "normal",
            },
            {
                "text": "계좌 정지 즉시 본인 인증 필요",
                "text_norm": "계좌 정지 즉시 본인 인증 필요",
                "label": "phishing",
            },
            {
                "text": "미납 요금 확인 후 즉시 입금",
                "text_norm": "미납 요금 확인 후 즉시 입금",
                "label": "phishing",
            },
            {
                "text": "보안 사고 발생 링크 확인",
                "text_norm": "보안 사고 발생 링크 확인 [URL]",
                "label": "phishing",
            },
        ]
    )


@pytest.fixture
def fitted_svm_model(
    svm_training_dataframe: pd.DataFrame,
) -> LinearSvmPhishingClassifier:
    """여러 테스트에서 재사용할 학습 완료 Linear SVM."""
    return LinearSvmPhishingClassifier().fit(
        svm_training_dataframe
    )


def test_fit_and_predict_decision_scores(
    fitted_svm_model: LinearSvmPhishingClassifier,
) -> None:
    """추론 결과가 유한한 1차원 decision score여야 합니다."""
    evaluation_df = pd.DataFrame(
        {
            "text_norm": [
                "내일 병원 예약 시간 안내",
                "계좌 정지 즉시 인증 [URL]",
            ]
        }
    )

    scores = fitted_svm_model.predict_scores(evaluation_df)
    predictions = fitted_svm_model.predict(evaluation_df)

    assert scores.score_type == ScoreType.DECISION
    assert scores.values.shape == (2,)
    assert np.isfinite(scores.values).all()
    assert set(predictions).issubset({"normal", "phishing"})


def test_default_threshold_is_zero() -> None:
    """decision score의 기본 threshold는 0이어야 합니다."""
    model = LinearSvmPhishingClassifier()

    assert model.default_threshold == 0.0


def test_predict_uses_text_norm_without_refitting_vectorizer(
    fitted_svm_model: LinearSvmPhishingClassifier,
) -> None:
    """추론은 학습된 vocabulary를 유지한 채 text_norm만 변환해야 합니다."""
    vocabulary_before = dict(
        fitted_svm_model.vectorizer.vocabulary_
    )

    evaluation_df = pd.DataFrame(
        {
            "text_norm": [
                "학습 데이터에 없던 새로운 문자열",
            ]
        }
    )

    fitted_svm_model.predict_scores(evaluation_df)

    assert fitted_svm_model.vectorizer.vocabulary_ == vocabulary_before


def test_phishing_score_direction_when_phishing_is_second_class(
    fitted_svm_model: LinearSvmPhishingClassifier,
) -> None:
    """classes_[1]이 phishing이면 decision score를 그대로 사용합니다."""
    fitted_svm_model.model.classes_ = np.asarray(
        ["normal", "phishing"]
    )

    raw_scores = np.asarray([-2.0, 0.0, 3.0])

    oriented_scores = fitted_svm_model._orient_phishing_scores(
        raw_scores
    )

    np.testing.assert_array_equal(
        oriented_scores,
        raw_scores,
    )


def test_phishing_score_direction_when_phishing_is_first_class(
    fitted_svm_model: LinearSvmPhishingClassifier,
) -> None:
    """classes_[0]이 phishing이면 decision score의 부호를 반전합니다."""
    fitted_svm_model.model.classes_ = np.asarray(
        ["phishing", "normal"]
    )

    raw_scores = np.asarray([-2.0, 0.0, 3.0])

    oriented_scores = fitted_svm_model._orient_phishing_scores(
        raw_scores
    )

    np.testing.assert_array_equal(
        oriented_scores,
        -raw_scores,
    )


def test_validation_threshold_can_be_selected(
    fitted_svm_model: LinearSvmPhishingClassifier,
    svm_validation_dataframe: pd.DataFrame,
) -> None:
    """validation score를 사용해 별도의 decision threshold를 선택합니다."""
    validation_scores = fitted_svm_model.predict_scores(
        svm_validation_dataframe
    )

    selection = select_validation_threshold(
        svm_validation_dataframe["label"].to_numpy(),
        validation_scores.values,
        target_recall=0.80,
    )

    predictions = fitted_svm_model.predict(
        svm_validation_dataframe,
        threshold=selection.threshold,
    )

    recall = recall_score(
        svm_validation_dataframe["label"],
        predictions,
        pos_label="phishing",
        zero_division=0,
    )

    assert np.isfinite(selection.threshold)
    assert selection.target_recall == 0.80
    assert recall >= 0.80


@pytest.mark.parametrize(
    "text_norm",
    [
        "",
        "   ",
        "!!! @@@ ###",
        "[URL] [PHONE] [ACCOUNT] [AMOUNT]",
    ],
)
def test_predict_handles_empty_special_and_masked_text(
    fitted_svm_model: LinearSvmPhishingClassifier,
    text_norm: str,
) -> None:
    """빈 문자열이나 특수 입력에서도 유한한 score를 반환해야 합니다."""
    scores = fitted_svm_model.predict_scores(
        pd.DataFrame(
            {
                "text_norm": [text_norm],
            }
        )
    )

    assert scores.values.shape == (1,)
    assert np.isfinite(scores.values[0])


def test_predict_before_fit_fails(
    svm_training_dataframe: pd.DataFrame,
) -> None:
    """fit 호출 전에는 decision score를 계산할 수 없어야 합니다."""
    model = LinearSvmPhishingClassifier()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_scores(svm_training_dataframe)


@pytest.mark.parametrize(
    ("dataframe", "error_type", "message"),
    [
        (
            pd.DataFrame({"text": ["일반 안내"]}),
            ValueError,
            "missing",
        ),
        (
            pd.DataFrame({"text_norm": []}),
            ValueError,
            "empty",
        ),
        (
            pd.DataFrame({"text_norm": [None]}),
            ValueError,
            "missing values",
        ),
        (
            pd.DataFrame({"text_norm": [123]}),
            TypeError,
            "must be strings",
        ),
    ],
)
def test_predict_validates_input_dataframe(
    fitted_svm_model: LinearSvmPhishingClassifier,
    dataframe: pd.DataFrame,
    error_type: type[Exception],
    message: str,
) -> None:
    """잘못된 추론 입력을 명확한 예외로 거부해야 합니다."""
    with pytest.raises(error_type, match=message):
        fitted_svm_model.predict_scores(dataframe)


def test_fit_requires_normal_and_phishing_labels() -> None:
    """학습 데이터에는 정상과 피싱 클래스가 모두 있어야 합니다."""
    model = LinearSvmPhishingClassifier()

    with pytest.raises(
        ValueError,
        match="both normal and phishing",
    ):
        model.fit(
            pd.DataFrame(
                {
                    "text_norm": [
                        "일반 안내입니다",
                        "가족 모임 안내입니다",
                    ],
                    "label": [
                        "normal",
                        "normal",
                    ],
                }
            )
        )


def test_joblib_round_trip_preserves_scores(
    fitted_svm_model: LinearSvmPhishingClassifier,
    tmp_path: Path,
) -> None:
    """저장하고 복원한 모델이 동일한 decision score를 반환해야 합니다."""
    evaluation_df = pd.DataFrame(
        {
            "text_norm": [
                "오늘 저녁 식사 약속 안내",
                "즉시 계좌 인증 [URL]",
                "[국외발신] [ACCOUNT] 입금 요청",
            ]
        }
    )

    expected_scores = fitted_svm_model.predict_scores(
        evaluation_df
    ).values

    artifact_path = tmp_path / "linear_svm.joblib"

    joblib.dump(
        fitted_svm_model,
        artifact_path,
    )

    restored_model = joblib.load(artifact_path)

    restored_scores = restored_model.predict_scores(
        evaluation_df
    ).values

    np.testing.assert_allclose(
        restored_scores,
        expected_scores,
        rtol=0.0,
        atol=0.0,
    )


def test_metadata_records_reproducible_configuration(
    fitted_svm_model: LinearSvmPhishingClassifier,
) -> None:
    """metadata에 TF-IDF와 Linear SVM 설정이 기록되어야 합니다."""
    metadata = fitted_svm_model.get_metadata()

    assert metadata["model_name"] == "linear_svm_char_tfidf"
    assert metadata["score_type"] == "decision"
    assert metadata["default_threshold"] == 0.0

    assert metadata["preprocessing"] == {
        "input_column": "text_norm",
        "normalization": (
            "app.analysis.text.preprocessing.normalize_text"
        ),
    }

    assert metadata["vectorizer"] == {
        "type": "TfidfVectorizer",
        "analyzer": "char_wb",
        "ngram_range": list(DEFAULT_NGRAM_RANGE),
        "min_df": DEFAULT_MIN_DF,
        "max_df": DEFAULT_MAX_DF,
        "max_features": DEFAULT_MAX_FEATURES,
        "lowercase": False,
        "sublinear_tf": True,
    }

    assert metadata["classifier"] == {
        "type": "LinearSVC",
        "C": DEFAULT_C,
        "class_weight": DEFAULT_CLASS_WEIGHT,
        "random_state": DEFAULT_RANDOM_STATE,
        "max_iter": DEFAULT_MAX_ITER,
        "dual": "auto",
    }