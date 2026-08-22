"""사전학습 한국어 인코더를 파인튜닝하는 피싱 분류기

기존 KoreanEncoderPhishingClassifier는 인코더를 완전히 얼린 채
mean pooling 임베딩 위에 LogisticRegression만 얹는다. 범용 임베딩이라
"기관 안내문"처럼 정상과 피싱의 표면이 거의 같은 구간을 가르지 못했다
(#102 진단: real_holdout에서 정상택배 78.6%, 정상공공기관 71.0% 오탐인 반면
어휘가 겹치지 않는 일상대화는 2.0%). 인코더까지 태스크에 맞춰 학습시켜
그 구간의 표현을 직접 다듬는다.
"""
from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from data_science.SMSModel.modeling.base import (
    BasePhishingClassifier,
    ScoreOutput,
    ScoreType,
)

DEFAULT_MODEL_ID = "beomi/KcELECTRA-small-v2022"

# frozen 버전과 동일한 값 - 벤치마크에서 p95 23.85ms, 절단 2.0%로 확인했다
DEFAULT_MAX_LENGTH = 256

DEFAULT_EPOCHS = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_RANDOM_STATE = 42

# MPS는 같은 시드로도 커널 구현에 따라 결과가 흔들려 다른 머신에서 재현되지
# 않는다. 학습 규모가 작아(train 692건, 16.5M 파라미터) CPU로도 전체 학습이
# 3분 이내라, 속도보다 재현성을 택한다.
DEFAULT_DEVICE = "cpu"

# 라벨을 정수로 고정해 fold마다 클래스 순서가 뒤집히지 않게 한다
LABEL_TO_INDEX: dict[str, int] = {"normal": 0, "phishing": 1}


class FineTunedKoreanEncoderClassifier(BasePhishingClassifier):
    """사전학습 인코더를 분류 헤드까지 end-to-end로 파인튜닝하는 분류기"""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_length: int = DEFAULT_MAX_LENGTH,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        random_state: int = DEFAULT_RANDOM_STATE,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_id = model_id
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

        # 파인튜닝된 가중치는 artifact에 함께 저장해야 하므로 상태로 보관한다
        # (frozen 버전은 사전학습 가중치를 그대로 다시 받아오면 그만이었다)
        self._weights: dict[str, Any] | None = None
        self._model = None
        self._tokenizer = None
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        """평가 보고서와 artifact에서 사용할 모델 이름"""
        return "korean_encoder_kcelectra_finetuned"

    @property
    def score_type(self) -> ScoreType:
        """분류 헤드의 softmax 확률을 반환한다"""
        return ScoreType.PROBABILITY

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
        *,
        require_label: bool,
    ) -> None:
        """학습 또는 추론에 필요한 DataFrame 구조와 값을 검증"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("input must be a pandas DataFrame")

        required_columns = {"text_norm"}

        if require_label:
            required_columns.add("label")

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"missing fine-tuned Korean encoder columns: {missing}"
            )

        if df.empty:
            raise ValueError(
                "fine-tuned Korean encoder input DataFrame is empty"
            )

        if df[list(required_columns)].isna().any().any():
            raise ValueError(
                "fine-tuned Korean encoder input contains missing values"
            )

        if not df["text_norm"].map(
            lambda value: isinstance(value, str)
        ).all():
            raise TypeError("text_norm values must be strings")

        if require_label:
            labels = set(df["label"].astype(str))
            supported_labels = set(LABEL_TO_INDEX)

            if not labels.issubset(supported_labels):
                raise ValueError(
                    f"unsupported labels: {labels - supported_labels}"
                )

            if labels != supported_labels:
                raise ValueError(
                    "training data must contain both normal and phishing labels"
                )

    def _seed_everything(self) -> None:
        """같은 random_state면 같은 가중치가 나오도록 난수원을 고정"""
        import random

        import torch

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _load_tokenizer(self):
        """토크나이저를 지연 로드. 파인튜닝해도 어휘는 그대로다"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        return self._tokenizer

    def _build_model(self):
        """사전학습 가중치 위에 새 분류 헤드를 얹은 모델을 생성"""
        from transformers import AutoModelForSequenceClassification

        return AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels=len(LABEL_TO_INDEX),
        )

    def _restored_model(self):
        """파인튜닝된 가중치를 적용한 추론용 모델을 지연 복원"""
        if self._model is None:
            if self._weights is None:
                raise RuntimeError(
                    "fine-tuned Korean encoder weights are unavailable"
                )

            model = self._build_model()
            model.load_state_dict(self._weights)
            model.to(self.device)
            model.eval()
            self._model = model

        return self._model

    def _class_weights(self, indices: np.ndarray):
        """정상이 피싱의 2배인 불균형을 손실에서 보정 (balanced와 같은 정의)"""
        import torch

        counts = np.bincount(indices, minlength=len(LABEL_TO_INDEX))

        if (counts == 0).any():
            raise ValueError(
                "training data must contain both normal and phishing labels"
            )

        weights = len(indices) / (len(LABEL_TO_INDEX) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def fit(self, train_df: pd.DataFrame) -> Self:
        """전달된 train split만 사용해 인코더까지 파인튜닝"""
        import torch
        from torch.nn import CrossEntropyLoss

        self._validate_dataframe(train_df, require_label=True)
        self._seed_everything()

        tokenizer = self._load_tokenizer()
        model = self._build_model()
        model.to(self.device)
        model.train()

        texts = train_df["text_norm"].astype(str).tolist()
        indices = (
            train_df["label"]
            .astype(str)
            .map(LABEL_TO_INDEX)
            .to_numpy(dtype=np.int64)
        )

        loss_function = CrossEntropyLoss(
            weight=self._class_weights(indices)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 셔플 순서까지 시드에 묶어 fold 재실행 시 같은 결과가 나오게 한다
        generator = np.random.default_rng(self.random_state)
        row_count = len(texts)

        for _ in range(self.epochs):
            order = generator.permutation(row_count)

            for start in range(0, row_count, self.batch_size):
                batch = order[start : start + self.batch_size]

                encoded = tokenizer(
                    [texts[index] for index in batch],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                targets = torch.tensor(
                    indices[batch],
                    dtype=torch.long,
                    device=self.device,
                )

                optimizer.zero_grad()
                logits = model(**encoded).logits
                loss_function(logits, targets).backward()
                optimizer.step()

        model.eval()

        # artifact에 담을 수 있도록 CPU 텐서로 내려 보관
        self._weights = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in model.state_dict().items()
        }
        self._model = model
        self._is_fitted = True
        return self

    def _require_fitted(self) -> None:
        """학습되지 않은 분류기의 사용을 차단"""
        if not self._is_fitted:
            raise RuntimeError("fine-tuned Korean encoder model is not fitted")

        if self._weights is None:
            raise RuntimeError(
                "fine-tuned Korean encoder weights are unavailable"
            )

    def predict_scores(self, df: pd.DataFrame) -> ScoreOutput:
        """각 메시지의 phishing 확률을 반환

        frozen 버전과 같은 이유로 한 건씩 처리한다. 묶어서 패딩하면 같은
        문자라도 배치 구성에 따라 결과가 미세하게 흔들리는데, 평가는 묶어서
        하고 운영은 단건으로 하는 구조라 그 차이가 재현성 문제가 된다.
        """
        import torch

        self._require_fitted()
        self._validate_dataframe(df, require_label=False)

        tokenizer = self._load_tokenizer()
        model = self._restored_model()
        phishing_index = LABEL_TO_INDEX["phishing"]

        probabilities: list[float] = []

        for value in df["text_norm"]:
            encoded = tokenizer(
                str(value),
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = model(**encoded).logits

            scores = torch.softmax(logits, dim=-1)[0]
            probabilities.append(float(scores[phishing_index]))

        return ScoreOutput(
            values=np.asarray(probabilities, dtype=float),
            score_type=self.score_type,
        )

    def __getstate__(self) -> dict[str, Any]:
        """파인튜닝된 가중치는 artifact에 반드시 포함한다

        frozen 버전은 사전학습 가중치를 다시 내려받으면 됐지만, 파인튜닝된
        가중치는 이 모델 자체라 빠지면 복원할 방법이 없다. 재구성 가능한
        모델 객체와 토크나이저만 제외한다.
        """
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """역직렬화 후 모델은 최초 사용 시 가중치로부터 다시 복원"""
        self.__dict__.update(state)

    def get_metadata(self) -> dict[str, Any]:
        """모델 재현과 artifact 추적에 필요한 설정을 반환"""
        return {
            **super().get_metadata(),
            "preprocessing": {
                "input_column": "text_norm",
                "normalization": (
                    "app.analysis.text.preprocessing.normalize_text"
                ),
            },
            "encoder": {
                "model_id": self.model_id,
                "max_length": self.max_length,
                "pooling": "classification_head",
                "weights_frozen": False,
            },
            "fine_tuning": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "device": self.device,
            },
        }
