import numpy as np
import pytest

from data_science.SMSModel.modeling import (
    NaiveBayesPhishingClassifier,
    ScoreType,
)

@pytest.mark.parametrize(
    "include_structural_features",
    [False, True],
)
def test_naive_bayes_fit_and_predict_scores(
    training_dataframe,
    include_structural_features,
):
    model = NaiveBayesPhishingClassifier(
        include_structural_features=(
            include_structural_features
        ),
        calibration_cv=2,
    )

    model.fit(training_dataframe)

    scores = model.predict_scores(
        training_dataframe.iloc[:4]
    )

    assert scores.score_type == (
        ScoreType.PROBABILITY
    )
    assert scores.values.shape == (4,)
    assert np.all(scores.values >= 0.0)
    assert np.all(scores.values <= 1.0)


def test_structural_model_has_six_additional_features(
    training_dataframe,
):
    text_only = NaiveBayesPhishingClassifier(
        include_structural_features=False,
        calibration_cv=2,
    )
    structural = NaiveBayesPhishingClassifier(
        include_structural_features=True,
        calibration_cv=2,
    )

    text_only.fit(training_dataframe)
    structural.fit(training_dataframe)

    text_only_matrix = (
        text_only._build_feature_matrix(
            training_dataframe,
            fit_vectorizer=False,
        )
    )
    structural_matrix = (
        structural._build_feature_matrix(
            training_dataframe,
            fit_vectorizer=False,
        )
    )

    assert (
        structural_matrix.shape[1]
        == text_only_matrix.shape[1] + 6
    )


def test_predict_before_fit_fails(
    training_dataframe,
):
    model = NaiveBayesPhishingClassifier(
        include_structural_features=True,
    )

    with pytest.raises(
        RuntimeError,
        match="not fitted",
    ):
        model.predict_scores(
            training_dataframe
        )
