"""배포 경로에 커밋된 stacking artifact가 현재 코드·데이터와 맞는지 검증한다.

다른 테스트가 합성 artifact로 코드 동작을 확인하는 것과 달리, 여기서는
저장소에 실제로 커밋된 파일을 연다. artifact는 코드와 따로 갱신되므로
둘이 어긋나도 조용히 지나가기 때문이다.

실제로 2026-08-11에 만든 프로덕션 artifact가 구조 특징 15개로 학습됐는데
이후 피처가 추가돼 코드가 20개를 만들면서, 추론이 항상 ValueError로
실패하고 있었다. 앱이 예외를 잡아 is_available=False로 처리하는 탓에
서비스는 죽지 않았고, 그래서 아무도 눈치채지 못한 채 모든 문자가 LLM으로
넘어가고 있었다. #107이 v9에서 고친 것과 같은 문제였다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest

from data_science.SMSModel.modeling.stacking import StackingPhishingClassifier
from data_science.SMSModel.run_stacking_training import (
    EXPECTED_DATASET_FINGERPRINT,
    PRODUCTION_ARTIFACT_DIRECTORY,
)

MODEL_PATH = PRODUCTION_ARTIFACT_DIRECTORY / "model.joblib"
METADATA_PATH = PRODUCTION_ARTIFACT_DIRECTORY / "metadata.json"

# 배포 이미지의 requirements.txt에 없는 패키지를 요구하는 base model.
# 인코더는 torch/transformers를 import하므로 이 artifact에 들어가면
# 배포 환경에서 로드 자체가 실패한다 (#101).
UNDEPLOYABLE_BASE_MODELS = frozenset({"korean_encoder"})


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def classifier() -> StackingPhishingClassifier:
    payload = joblib.load(MODEL_PATH)
    return payload["classifier"]


def test_artifact_files_exist() -> None:
    """배포 이미지가 이 두 파일을 그대로 복사한다"""
    assert MODEL_PATH.is_file(), f"프로덕션 artifact 없음: {MODEL_PATH}"
    assert METADATA_PATH.is_file(), f"metadata 없음: {METADATA_PATH}"


def test_checksum_matches_the_committed_model(metadata: dict) -> None:
    """앱이 로드 전에 이 값을 대조하므로 어긋나면 모델을 아예 못 쓴다"""
    digest = hashlib.sha256()

    with MODEL_PATH.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(8192), b""):
            digest.update(chunk)

    assert digest.hexdigest() == metadata.get("model_sha256"), (
        "metadata의 model_sha256이 실제 파일과 다르다. "
        "둘 중 하나만 커밋됐을 가능성이 높다."
    )


def test_inference_works_with_the_current_code(
    classifier: StackingPhishingClassifier,
) -> None:
    """학습 시 피처 수와 현재 코드가 만드는 피처 수가 같아야 한다.

    이 테스트가 없어서 프로덕션 artifact가 몇 달간 고장난 채 방치됐다.
    구조 특징을 추가·제외할 때마다 artifact를 다시 만들어야 한다.
    """
    prediction = classifier.predict_one("오늘 저녁 같이 먹자")

    assert 0.0 <= prediction.risk_probability <= 1.0
    assert prediction.unavailable_models == ()


def test_meta_classifier_input_width_matches_the_code(
    classifier: StackingPhishingClassifier,
) -> None:
    """predict_one이 잡아내지만, 실패 원인을 바로 읽히게 따로 남긴다"""
    expected = len(classifier.feature_names)
    actual = classifier.meta_classifier.coef_.shape[1]

    assert actual == expected, (
        f"artifact는 피처 {actual}개로 학습됐는데 현재 코드는 {expected}개를 "
        "만든다. --deployable로 재학습해야 한다."
    )


def test_trained_on_the_current_dataset(metadata: dict) -> None:
    """데이터셋이 바뀌면 artifact도 다시 만들어야 한다.

    fingerprint가 어긋나면 정리·보강 이전 데이터로 학습된 모델이 배포된다.
    동작은 하므로 테스트 없이는 드러나지 않는다.
    """
    actual = metadata.get("dataset", {}).get("dataset_fingerprint")

    assert actual == EXPECTED_DATASET_FINGERPRINT, (
        "프로덕션 artifact가 현재 데이터셋으로 학습되지 않았다. "
        "--deployable로 재학습해야 한다."
    )


def test_contains_no_undeployable_base_model(
    classifier: StackingPhishingClassifier,
    metadata: dict,
) -> None:
    """배포 이미지에 없는 의존성을 요구하는 모델이 들어가면 안 된다.

    인코더는 torch/transformers를 import하는데 배포 requirements에는
    없다(#101). 연구용 artifact를 실수로 이 경로에 올리는 것을 막는다.
    """
    included = set(classifier.base_models) | set(
        metadata.get("model", {}).get("base_models", [])
    )
    blocked = included & UNDEPLOYABLE_BASE_MODELS

    assert not blocked, (
        f"배포 불가 base model이 포함돼 있다: {sorted(blocked)}. "
        "프로덕션 artifact는 --deployable로 만들어야 한다."
    )


def test_app_loads_the_artifact_through_its_own_path() -> None:
    """앱이 실제로 쓰는 경로와 검증 절차로 한 번 더 확인한다"""
    from app.analysis.text import stacking_analyzer

    assert stacking_analyzer.DEFAULT_STACKING_MODEL_PATH == MODEL_PATH

    result = stacking_analyzer.analyze_text_with_stacking("오늘 저녁 같이 먹자")

    assert result["is_available"] is True, (
        "앱 경로로 로드하거나 추론하는 데 실패했다: "
        f"{result['result'].get('error_message')}"
    )
