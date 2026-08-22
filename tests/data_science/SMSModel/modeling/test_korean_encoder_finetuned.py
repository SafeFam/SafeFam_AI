"""파인튜닝 한국어 인코더 분류기 테스트

실제 인코더를 내려받거나 학습시키지 않도록 모델과 토크나이저를 결정적
스텁으로 대체한다. 확인하는 것은 표현 품질이 아니라 Adapter 계약과
직렬화 규약이다 - 특히 frozen 버전과 정반대로 "가중치가 artifact에
반드시 실려야 한다"는 점이다.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from data_science.SMSModel.modeling.base import ScoreType
from data_science.SMSModel.modeling.korean_encoder_finetuned import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    LABEL_TO_INDEX,
    FineTunedKoreanEncoderClassifier,
)
from data_science.SMSModel.modeling.stacking import (
    _default_base_model_factories,
)

PHISHING_MARKERS = ("급등", "종목", "입장", "인증번호")

# 스텁 토크나이저가 만드는 입력 차원
_FEATURE_SIZE = len(PHISHING_MARKERS) + 1


class _StubTokenizer:
    """마커 포함 여부를 그대로 특징 벡터로 만드는 결정적 토크나이저"""

    def __call__(
        self,
        text,
        *,
        truncation=True,
        max_length=None,
        padding=False,
        return_tensors="pt",
    ):
        texts = [text] if isinstance(text, str) else list(text)
        rows = []

        for value in texts:
            hits = [float(marker in str(value)) for marker in PHISHING_MARKERS]
            rows.append(hits + [float(len(str(value)) % 7) / 7.0])

        return _StubEncoding(
            {"features": torch.tensor(rows, dtype=torch.float32)}
        )


class _StubEncoding(dict):
    """transformers의 BatchEncoding처럼 .to()를 지원하는 최소 구현"""

    def to(self, device):
        return _StubEncoding(
            {key: value.to(device) for key, value in self.items()}
        )


class _StubOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _StubModel(torch.nn.Module):
    """실제 인코더 대신 학습 가능한 선형 계층 하나만 둔 모델"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_FEATURE_SIZE, len(LABEL_TO_INDEX))

    def forward(self, *, features: torch.Tensor) -> _StubOutput:
        return _StubOutput(self.linear(features))


@pytest.fixture
def classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> FineTunedKoreanEncoderClassifier:
    """인코더를 내려받지 않는 분류기"""
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_build_model",
        lambda self: _StubModel(),
    )
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_load_tokenizer",
        lambda self: _StubTokenizer(),
    )

    # 스텁은 선형 계층 하나라, 실제 인코더용 학습률(2e-5)로는 몇 스텝 안에
    # 아무것도 학습되지 않는다. 스텁 규모에 맞춰 키운다.
    return FineTunedKoreanEncoderClassifier(
        epochs=30,
        learning_rate=0.1,
    )


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


@pytest.fixture
def scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_norm": [
                "급등 종목 무료 공개 오픈채팅 입장 코드",
                "오늘 가족 모임 시간을 안내합니다",
            ]
        }
    )


def test_scores_are_phishing_probabilities(
    classifier: FineTunedKoreanEncoderClassifier,
    training_dataframe: pd.DataFrame,
    scored: pd.DataFrame,
) -> None:
    """확률 방향이 phishing 쪽이어야 meta-classifier가 부호를 오해하지 않는다"""
    classifier.fit(training_dataframe)

    output = classifier.predict_scores(scored)

    assert output.score_type == ScoreType.PROBABILITY
    assert ((output.values >= 0.0) & (output.values <= 1.0)).all()
    assert output.values[0] > output.values[1]


def test_fine_tuned_weights_are_serialized(
    classifier: FineTunedKoreanEncoderClassifier,
    training_dataframe: pd.DataFrame,
    scored: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """frozen 버전과 정반대로, 파인튜닝된 가중치가 빠지면 복원할 방법이 없다.

    사전학습 가중치는 다시 내려받으면 그만이지만 파인튜닝 결과는 이 모델
    자체다. artifact 왕복 후에도 같은 확률이 나와야 한다.
    """
    classifier.fit(training_dataframe)
    expected = classifier.predict_scores(scored).values

    artifact_path = tmp_path / "encoder.joblib"
    joblib.dump(classifier, artifact_path)
    restored = joblib.load(artifact_path)

    assert restored._weights is not None
    assert restored._model is None
    assert restored._tokenizer is None

    np.testing.assert_array_equal(
        restored.predict_scores(scored).values,
        expected,
    )


def test_training_is_reproducible_for_the_same_seed(
    monkeypatch: pytest.MonkeyPatch,
    training_dataframe: pd.DataFrame,
    scored: pd.DataFrame,
) -> None:
    """같은 random_state면 같은 확률이 나와야 artifact를 재현할 수 있다"""
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_build_model",
        lambda self: _StubModel(),
    )
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_load_tokenizer",
        lambda self: _StubTokenizer(),
    )

    first = FineTunedKoreanEncoderClassifier(
        epochs=30, learning_rate=0.1, random_state=7
    )
    second = FineTunedKoreanEncoderClassifier(
        epochs=30, learning_rate=0.1, random_state=7
    )

    first.fit(training_dataframe)
    second.fit(training_dataframe)

    np.testing.assert_array_equal(
        first.predict_scores(scored).values,
        second.predict_scores(scored).values,
    )


def test_predict_before_fit_is_rejected(
    classifier: FineTunedKoreanEncoderClassifier,
) -> None:
    """학습 전 추론은 조용히 통과하면 안 된다"""
    with pytest.raises(RuntimeError):
        classifier.predict_scores(pd.DataFrame({"text_norm": ["안녕하세요"]}))


def test_training_requires_both_labels(
    classifier: FineTunedKoreanEncoderClassifier,
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
    classifier: FineTunedKoreanEncoderClassifier,
) -> None:
    """다른 base model과 같은 입력 컬럼을 요구해야 한다"""
    with pytest.raises(ValueError):
        classifier.fit(
            pd.DataFrame({"text": ["급등 종목"], "label": ["phishing"]})
        )


def test_metadata_records_the_fine_tuning_settings(
    classifier: FineTunedKoreanEncoderClassifier,
) -> None:
    """재현에 필요한 값은 artifact metadata에 남아야 한다"""
    metadata = classifier.get_metadata()

    assert metadata["model_name"] == "korean_encoder_kcelectra_finetuned"
    assert metadata["encoder"]["model_id"] == DEFAULT_MODEL_ID
    assert metadata["encoder"]["max_length"] == DEFAULT_MAX_LENGTH
    assert metadata["encoder"]["weights_frozen"] is False
    assert metadata["fine_tuning"]["epochs"] == 30
    assert metadata["fine_tuning"]["class_weight"] == "balanced"


def test_constant_learning_rate_is_the_default(
    classifier: FineTunedKoreanEncoderClassifier,
) -> None:
    """스케줄러 도입이 기존 artifact의 재현성을 깨면 안 된다.

    warmup_ratio 기본값 0은 스케줄러를 아예 만들지 않아 고정 학습률로
    학습하던 이전 동작을 그대로 유지한다.
    """
    assert classifier.warmup_ratio == 0.0
    assert classifier._build_scheduler(object(), row_count=100) is None
    assert classifier.get_metadata()["fine_tuning"]["lr_schedule"] == "constant"


def test_warmup_schedule_ramps_up_then_decays(
    monkeypatch: pytest.MonkeyPatch,
    training_dataframe: pd.DataFrame,
) -> None:
    """warmup 구간에서 학습률이 0에서 올라간 뒤 끝에서 0으로 내려와야 한다"""
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_build_model",
        lambda self: _StubModel(),
    )
    monkeypatch.setattr(
        FineTunedKoreanEncoderClassifier,
        "_load_tokenizer",
        lambda self: _StubTokenizer(),
    )

    classifier = FineTunedKoreanEncoderClassifier(
        epochs=4,
        batch_size=10,
        learning_rate=0.1,
        warmup_ratio=0.25,
    )
    optimizer = torch.optim.AdamW(_StubModel().parameters(), lr=0.1)
    scheduler = classifier._build_scheduler(optimizer, row_count=40)

    assert scheduler is not None

    # 4 epoch x 4 step = 16 step, warmup은 앞 4 step
    rates = []
    for _ in range(16):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[0] == pytest.approx(0.0)
    assert rates[4] == pytest.approx(0.1)
    assert rates[4] > rates[2] > rates[0]
    assert rates[-1] < rates[4]
    assert (
        classifier.get_metadata()["fine_tuning"]["lr_schedule"]
        == "linear_warmup_then_linear_decay"
    )


def test_invalid_warmup_ratio_is_rejected() -> None:
    """1 이상이면 학습 내내 warmup만 하다 끝난다"""
    with pytest.raises(ValueError):
        FineTunedKoreanEncoderClassifier(warmup_ratio=1.0)

    with pytest.raises(ValueError):
        FineTunedKoreanEncoderClassifier(warmup_ratio=-0.1)


def test_registered_as_a_stacking_base_model() -> None:
    """factory에 등록되지 않으면 학습에 전혀 참여하지 않는다"""
    factories = _default_base_model_factories()

    assert "korean_encoder" in factories
    assert isinstance(
        factories["korean_encoder"](),
        FineTunedKoreanEncoderClassifier,
    )


def test_default_epochs_matches_the_validation_selection() -> None:
    """epoch은 validation에서 시드 3개로 골랐다 (#102).

    4/6/8/10 비교에서 10이 모든 시드·모든 지표에서 우세했다. 8에서 한 번
    지표가 내려앉지만 시드를 바꾸면 재현되지 않는 단발성 변동이라, 그
    하락을 과학습으로 오해해 6으로 되돌리지 않도록 값을 고정한다.
    """
    assert FineTunedKoreanEncoderClassifier().epochs == 10


def test_cpu_is_the_default_device() -> None:
    """MPS/CUDA는 커널 구현에 따라 결과가 흔들려 다른 머신에서 재현되지 않는다.

    학습 규모가 작아 CPU로도 전체 학습이 수 분이라, 속도보다 재현성을 택한
    결정을 회귀 테스트로 고정한다.
    """
    assert FineTunedKoreanEncoderClassifier().device == "cpu"
