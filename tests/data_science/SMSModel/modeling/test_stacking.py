"""OOF stacking 분류기 테스트"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)
from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
    normalize_base_scores,
)

class KeywordClassifier(BasePhishingClassifier):
    """테스트에서만 사용하는 결정론적 가짜 모델"""

    def __init__(self, *, name: str, fail: bool = False) -> None:
        self._name = name
        self.fail = fail
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.PROBABILITY

    def fit(self, train_df: pd.DataFrame) -> "KeywordClassifier":
        self._is_fitted = True
        return self

    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        if self.fail:
            raise RuntimeError("expected model failure")

        if not self._is_fitted:
            raise RuntimeError("model is not fitted")

        values = df["text"].map(
            lambda text: 0.9 if "송금" in text else 0.1
        ).to_numpy(dtype=float)

        return ScoreOutput(
            values=values,
            score_type=ScoreType.PROBABILITY,
        )


@pytest.fixture
def stacking_training_dataframe() -> pd.DataFrame:
    rows = []

    # StratifiedGroupKFold가 동작하도록 클래스별 독립 그룹을 만듭니다.
    for index in range(15):
        rows.append(
            {
                "text": f"오늘 회의 시간 안내 {index}",
                "text_norm": f"오늘 회의 시간 안내 {index}",
                "has_url": False,
                "label": "normal",
                "template_group_id": f"normal-{index}",
            }
        )
        rows.append(
            {
                "text": f"즉시 계좌로 송금하세요 {index}",
                "text_norm": f"즉시 계좌로 송금하세요 {index}",
                "has_url": False,
                "label": "phishing",
                "template_group_id": f"phishing-{index}",
            }
        )

    return pd.DataFrame(rows)


def build_factories(*, fail_second: bool = False):
    return {
        "model_a": lambda: KeywordClassifier(name="model_a"),
        "model_b": lambda: KeywordClassifier(
            name="model_b",
            fail=fail_second,
        ),
        "model_c": lambda: KeywordClassifier(name="model_c"),
    }


def test_normalizes_decision_scores() -> None:
    output = ScoreOutput(
        values=np.asarray([-10.0, 0.0, 10.0]),
        score_type=ScoreType.DECISION,
    )

    normalized = normalize_base_scores(output)

    assert normalized[0] < 0.5
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] > 0.5


def test_fits_with_oof_predictions(
    stacking_training_dataframe: pd.DataFrame,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        base_model_factories=build_factories(),
    )

    classifier.fit(stacking_training_dataframe)

    normal = classifier.predict_one("오늘 저녁 같이 먹자")
    phishing = classifier.predict_one("즉시 계좌로 송금하세요")

    assert normal.risk_probability < phishing.risk_probability
    assert set(phishing.model_scores) == {
        "model_a",
        "model_b",
        "model_c",
    }


def test_reports_failed_base_model(
    stacking_training_dataframe: pd.DataFrame,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        base_model_factories=build_factories(),
    )
    classifier.fit(stacking_training_dataframe)

    classifier.base_models["model_b"].fail = True

    prediction = classifier.predict_one("즉시 계좌로 송금하세요")

    assert prediction.unavailable_models == ("model_b",)
    assert prediction.model_scores["model_b"] == pytest.approx(0.5)
    assert prediction.risk_probability >= 0.9


def test_rejects_training_without_groups(
    stacking_training_dataframe: pd.DataFrame,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        base_model_factories=build_factories(),
    )

    without_groups = stacking_training_dataframe.drop(
        columns=["template_group_id"]
    )

    with pytest.raises(ValueError, match="template_group_id"):
        classifier.fit(without_groups)


def test_prediction_is_reproducible(
    stacking_training_dataframe: pd.DataFrame,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        random_state=42,
        base_model_factories=build_factories(),
    )
    classifier.fit(stacking_training_dataframe)

    first = classifier.predict_one("즉시 계좌로 송금하세요")
    second = classifier.predict_one("즉시 계좌로 송금하세요")

    assert first == second


def test_predict_one_runs_each_base_model_once(
    stacking_training_dataframe: pd.DataFrame,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        base_model_factories=build_factories(),
    )
    classifier.fit(stacking_training_dataframe)

    call_counts: dict[str, int] = {}

    for name, model in classifier.base_models.items():
        original_predict_scores = model.predict_scores

        def counted_predict_scores(
            df: pd.DataFrame,
            *,
            model_name: str = name,
            predict=original_predict_scores,
        ) -> ScoreOutput:
            call_counts[model_name] = call_counts.get(model_name, 0) + 1
            return predict(df)

        model.predict_scores = counted_predict_scores  # type: ignore[method-assign]

    classifier.predict_one("즉시 계좌로 송금하세요")

    assert call_counts == {
        "model_a": 1,
        "model_b": 1,
        "model_c": 1,
    }


def test_default_classifier_is_joblib_serializable(
    stacking_training_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    classifier = StackingPhishingClassifier(n_splits=3)
    classifier.fit(stacking_training_dataframe)
    artifact_path = tmp_path / "stacking.joblib"
    prediction_input = "즉시 계좌로 송금하세요"
    original = classifier.predict_one(prediction_input)

    joblib.dump(classifier, artifact_path)
    loaded = joblib.load(artifact_path)

    reloaded = loaded.predict_one(prediction_input)
    assert reloaded.risk_probability == pytest.approx(
        original.risk_probability
    )
    assert reloaded.confidence == pytest.approx(original.confidence)
    assert reloaded.risk_score == original.risk_score
    assert (
        reloaded.is_suspected_phishing
        is original.is_suspected_phishing
    )


@pytest.mark.parametrize(
    ("probability", "expected_confidence"),
    [
        (0.1, 0.5),
        (0.6, 0.5),
    ],
)
def test_confidence_is_measured_from_active_threshold(
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
    expected_confidence: float,
) -> None:
    classifier = StackingPhishingClassifier(
        n_splits=3,
        threshold=0.2,
        base_model_factories=build_factories(),
    )
    monkeypatch.setattr(
        classifier,
        "_predict_probability_details",
        lambda _df: (
            np.asarray([probability]),
            (),
            {
                name: np.asarray([probability])
                for name in classifier.model_names
            },
        ),
    )

    prediction = classifier.predict_one("confidence branch test")

    assert prediction.confidence == pytest.approx(expected_confidence)
