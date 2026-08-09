"""모델 Adapter에서 기존 운영 API용 artifact를 저장하는 기능"""

from __future__ import annotations

from pathlib import Path

import joblib

from .naive_bayes import (
    NaiveBayesPhishingClassifier,
)


def save_operational_naive_bayes_artifacts(
    classifier: NaiveBayesPhishingClassifier,
    *,
    threshold: float,
    model_path: Path,
    vectorizer_path: Path,
) -> None:
    """structural NB 모델을 기존 운영 artifact 형식으로 저장"""

    classifier._require_fitted()

    if not classifier.include_structural_features:
        raise ValueError(
            "only the structural Naive Bayes model "
            "can be saved as the operational artifact"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability threshold must be between 0 and 1")

    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    vectorizer_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 기존 API와 완전히 같은 key만 저장
    joblib.dump(
        {
            "model": classifier.model,
            "threshold": float(threshold),
            "classes": list(classifier.classes_),
        },
        model_path,
    )

    # 기존 API는 vectorizer를 별도 pkl로 읽음
    joblib.dump(
        classifier.vectorizer,
        vectorizer_path,
    )
