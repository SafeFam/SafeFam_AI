"""사전학습 한국어 인코더 분류기 테스트

인코더 가중치를 내려받지 않도록 embed를 결정적 스텁으로 대체한다. 테스트가
확인하는 것은 임베딩 품질이 아니라 Adapter 계약과 직렬화 규약이다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data_science.SMSModel.modeling.base import ScoreType
from data_science.SMSModel.modeling.korean_encoder import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    KoreanEncoderPhishingClassifier,
)
from data_science.SMSModel.modeling.stacking import (
    _default_base_model_factories,
)

PHISHING_MARKERS = ("급등", "종목", "입장", "인증번호")


def _stub_embed(texts: pd.Series) -> np.ndarray:
    """마커 포함 여부로 두 무리를 갈라놓는 결정적 임베딩"""
    rows = []
    for value in texts:
        text = str(value)
        hits = [float(marker in text) for marker in PHISHING_MARKERS]
        rows.append(hits + [float(len(text) % 7) / 7.0])

    return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def classifier(monkeypatch: pytest.MonkeyPatch) -> KoreanEncoderPhishingClassifier:
    """인코더를 내려받지 않는 분류기"""
    monkeypatch.setattr(
        KoreanEncoderPhishingClassifier,
        "embed",
        lambda self, texts: _stub_embed(texts),
    )

    return KoreanEncoderPhishingClassifier()


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    """양쪽 label이 모두 들어간 균형 학습 데이터"""
    rows: list[dict[str, str]] = []

    for index in range(20):
        rows.append(
            {
                "text_norm": f"오늘 가족 모임 시간을 안내합니다 {index}",
                "label": "normal",
            }
        )
        rows.append(
            {
                "text_norm": f"급등 종목 무료 공개 오픈채팅 입장 {index}",
                "label": "phishing",
            }
        )

    return pd.DataFrame(rows)


def test_scores_are_phishing_probabilities(
    classifier: KoreanEncoderPhishingClassifier,
    training_dataframe: pd.DataFrame,
) -> None:
    """확률 방향이 phishing 쪽이어야 meta-classifier가 부호를 오해하지 않는다"""
    classifier.fit(training_dataframe)

    scored = pd.DataFrame(
        {
            "text_norm": [
                "급등 종목 무료 공개 오픈채팅 입장 코드",
                "오늘 가족 모임 시간을 안내합니다",
            ]
        }
    )
    output = classifier.predict_scores(scored)

    assert output.score_type == ScoreType.PROBABILITY
    assert ((output.values >= 0.0) & (output.values <= 1.0)).all()
    assert output.values[0] > output.values[1]


def test_predict_before_fit_is_rejected(
    classifier: KoreanEncoderPhishingClassifier,
) -> None:
    """학습 전 추론은 조용히 통과하면 안 된다"""
    with pytest.raises(RuntimeError):
        classifier.predict_scores(pd.DataFrame({"text_norm": ["안녕하세요"]}))


def test_training_requires_both_labels(
    classifier: KoreanEncoderPhishingClassifier,
) -> None:
    """한쪽 label만으로 학습하면 확률이 상수가 된다"""
    single_label = pd.DataFrame(
        {
            "text_norm": ["급등 종목 무료 공개", "오픈채팅 입장 코드"],
            "label": ["phishing", "phishing"],
        }
    )

    with pytest.raises(ValueError):
        classifier.fit(single_label)


def test_missing_text_norm_is_rejected(
    classifier: KoreanEncoderPhishingClassifier,
) -> None:
    """다른 base model과 같은 입력 컬럼을 요구해야 한다"""
    with pytest.raises(ValueError):
        classifier.fit(
            pd.DataFrame({"text": ["급등 종목"], "label": ["phishing"]})
        )


def test_encoder_weights_are_not_serialized(
    classifier: KoreanEncoderPhishingClassifier,
    training_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """인코더 가중치가 artifact에 실리면 저장소가 66MB씩 불어난다"""
    classifier.fit(training_dataframe)
    classifier._encoder = object()
    classifier._tokenizer = object()

    artifact_path = tmp_path / "encoder.joblib"
    joblib.dump(classifier, artifact_path)
    restored = joblib.load(artifact_path)

    assert restored._encoder is None
    assert restored._tokenizer is None
    assert restored.model_id == DEFAULT_MODEL_ID
    assert restored._is_fitted


def test_registered_as_a_stacking_base_model() -> None:
    """factory에 등록되지 않으면 학습에 전혀 참여하지 않는다"""
    factories = _default_base_model_factories()

    assert "korean_encoder" in factories
    assert isinstance(
        factories["korean_encoder"](),
        KoreanEncoderPhishingClassifier,
    )


def test_metadata_records_the_encoder_settings(
    classifier: KoreanEncoderPhishingClassifier,
) -> None:
    """재현에 필요한 값은 artifact metadata에 남아야 한다"""
    metadata = classifier.get_metadata()

    assert metadata["model_name"] == "korean_encoder_kcelectra"
    assert metadata["encoder"]["model_id"] == DEFAULT_MODEL_ID
    assert metadata["encoder"]["max_length"] == DEFAULT_MAX_LENGTH
    assert metadata["encoder"]["weights_frozen"] is True
    assert metadata["encoder"]["pooling"] == "mean"
