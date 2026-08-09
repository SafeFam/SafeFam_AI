"""모델 Adapter에서 기존 운영 API용 artifact를 저장하는 기능"""

from __future__ import annotations

import json
import os
import uuid
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
    """두 운영 artifact를 한 세대로 저장하고 포인터를 원자적으로 교체"""

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
    if model_path.parent.resolve() != vectorizer_path.parent.resolve():
        raise ValueError("model and vectorizer artifacts must share a directory")

    artifact_root = model_path.parent
    artifact_root.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    version_dir = artifact_root / "versions" / generation
    version_dir.mkdir(parents=True, exist_ok=False)
    versioned_model_path = version_dir / model_path.name
    versioned_vectorizer_path = version_dir / vectorizer_path.name

    # 기존 API와 완전히 같은 key만 저장
    joblib.dump(
        {
            "model": classifier.model,
            "threshold": float(threshold),
            "classes": list(classifier.classes_),
        },
        versioned_model_path,
    )

    # 기존 API는 vectorizer를 별도 pkl로 읽음
    joblib.dump(
        classifier.vectorizer,
        versioned_vectorizer_path,
    )

    # 포인터를 바꾸기 전에 두 파일이 모두 정상적으로 역직렬화되는지 확인합니다.
    joblib.load(versioned_model_path)
    joblib.load(versioned_vectorizer_path)

    pointer_path = artifact_root / "current.json"
    temporary_pointer_path = artifact_root / f".current-{generation}.tmp"
    temporary_pointer_path.write_text(
        json.dumps(
            {
                "generation": generation,
                "model": model_path.name,
                "vectorizer": vectorizer_path.name,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_pointer_path, pointer_path)
