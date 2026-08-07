"""Naive Bayes 운영 artifact와 기존 API 호환 테스트."""

import joblib
import pytest

from app.analysis.text import naive_bayes_analyzer as api_analyzer
from data_science.SMSModel.modeling import (
    NaiveBayesPhishingClassifier,
    save_operational_naive_bayes_artifacts,
)


@pytest.fixture(autouse=True)
def reset_api_model_cache():
    """테스트별로 기존 API의 모듈 전역 모델 캐시를 초기화합니다."""
    for name, value in (
        ("_model", None),
        ("_vectorizer", None),
        ("_threshold", None),
        ("_classes", None),
        ("_load_error", None),
        ("_load_attempted", False),
    ):
        setattr(api_analyzer, name, value)
    yield
    api_analyzer._model = None
    api_analyzer._vectorizer = None
    api_analyzer._threshold = None
    api_analyzer._classes = None
    api_analyzer._load_error = None
    api_analyzer._load_attempted = False


def test_structural_artifact_keeps_existing_api_schema(
    tmp_path,
    training_dataframe,
):
    classifier = NaiveBayesPhishingClassifier(
        include_structural_features=True,
        calibration_cv=2,
    ).fit(training_dataframe)
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"

    save_operational_naive_bayes_artifacts(
        classifier,
        threshold=0.4,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )

    artifact = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    assert set(artifact) == {"model", "threshold", "classes"}
    assert artifact["threshold"] == 0.4
    assert artifact["classes"] == ["normal", "phishing"]
    assert hasattr(artifact["model"], "predict_proba")
    assert hasattr(vectorizer, "transform")


@pytest.mark.asyncio
async def test_saved_structural_artifact_loads_in_existing_api(
    monkeypatch,
    tmp_path,
    training_dataframe,
):
    classifier = NaiveBayesPhishingClassifier(
        include_structural_features=True,
        calibration_cv=2,
    ).fit(training_dataframe)
    model_path = tmp_path / "model.pkl"
    vectorizer_path = tmp_path / "vectorizer.pkl"
    save_operational_naive_bayes_artifacts(
        classifier,
        threshold=0.4,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )
    monkeypatch.setattr(api_analyzer, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_analyzer, "VECTORIZER_PATH", vectorizer_path)

    result = await api_analyzer.analyze_text_with_naive_bayes(
        "계좌 정지 확인 필요 http://bit.ly/fake99"
    )

    assert result["engine"] == "naive_bayes"
    assert result["is_available"] is True
    assert 0 <= result["result"]["risk_score"] <= 100
    assert result["result"]["error_message"] is None


def test_operational_artifact_rejects_text_only_model(
    tmp_path,
    training_dataframe,
):
    classifier = NaiveBayesPhishingClassifier(
        include_structural_features=False,
        calibration_cv=2,
    ).fit(training_dataframe)

    with pytest.raises(ValueError, match="structural"):
        save_operational_naive_bayes_artifacts(
            classifier,
            threshold=0.4,
            model_path=tmp_path / "model.pkl",
            vectorizer_path=tmp_path / "vectorizer.pkl",
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_operational_artifact_rejects_invalid_probability_threshold(
    tmp_path,
    training_dataframe,
    threshold,
):
    classifier = NaiveBayesPhishingClassifier(
        include_structural_features=True,
        calibration_cv=2,
    ).fit(training_dataframe)

    with pytest.raises(ValueError, match="between 0 and 1"):
        save_operational_naive_bayes_artifacts(
            classifier,
            threshold=threshold,
            model_path=tmp_path / "model.pkl",
            vectorizer_path=tmp_path / "vectorizer.pkl",
        )
