"""학습된 자체 모델을 사용하는 추론 서비스"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import joblib

from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)


logger = logging.getLogger(__name__)

DEFAULT_STACKING_MODEL_PATH = Path(
    "data_science/SMSModel/artifacts/stacking/model.joblib"
)

_load_lock = threading.Lock()
_load_attempted = False
_classifier: StackingPhishingClassifier | None = None
_load_error: str | None = None

def _load_classifier(
    model_path: Path = DEFAULT_STACKING_MODEL_PATH,    
) -> None:
    """artifact를 프로세스당 한 번만 안전하게 로드"""

    global _load_attempted
    global _classifier
    global _load_error

    if _load_attempted:
        return

    with _load_lock:
        if _load_attempted:
            return

        try:
            payload = joblib.load(model_path)

            if not isinstance(payload, dict):
                raise TypeError("stacking artifact must be a dictionary")

            if payload.get("schema_version") != 1:
                raise ValueError("unsupported stacking artifact schema")

            classifier = payload.get("classifier")

            if not isinstance(
                classifier,
                StackingPhishingClassifier,
            ):
                raise TypeError(
                    "artifact does not contain a stacking classifier"
                )

            # 모든 검증이 끝난 뒤 전역 상태 반영
            _classifier = classifier
            _load_error = None

            logger.info(
                "[Stacking] 모델 로드 완료 (threshold=%s)",
                classifier.threshold,
            )
        except Exception as exception:  # noqa: BLE001
            _classifier = None
            _load_error = type(exception).__name__

            logger.error(
                "[Stacking] 모델 로드 실패. error_type=%s",
                _load_error,
            )
        finally:
            _load_attempted = True

def is_stacking_model_loaded() -> bool:
    """stacking 모델 사용 가능 여부 반환"""

    _load_classifier()
    return _classifier is not None

def analyze_text_with_stacking(text: str) -> dict[str, Any]:
    """자체 모델의 위험 점수, 신뢰도와 단계별 점수 반환"""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    _load_classifier()

    if _classifier is None:
        # 로드 실패를 정상 또는 0점으로 표현 X
        return {
            "engine": "stacking",
            "is_available": False,
            "result": {
                "risk_score": None,
                "risk_probability": None,
                "confidence": 0.0,
                "is_suspected_phishing": None,
                "threshold": None,
                "model_scores": {},
                "unavailable_models": [],
                "error_message": "Stacking Model Unavailable",
            },
        }

    try:
        prediction = _classifier.predict_one(text)

        logger.info(
            "[Stacking] 분석 완료 - risk_score=%s confidence=%.4f "
            "unavailable_model_count=%s",
            prediction.risk_score,
            prediction.confidence,
            len(prediction.unavailable_models),
        )

        return {
            "engine": "stacking",
            "is_available": True,
            "result": {
                "risk_score": prediction.risk_score,
                "risk_probability": prediction.risk_probability,
                "confidence": prediction.confidence,
                "is_suspected_phishing": (
                    prediction.is_suspected_phishing
                ),
                "threshold": prediction.threshold,
                "model_scores": prediction.model_scores,
                "unavailable_models": list(
                    prediction.unavailable_models
                ),
                "error_message": None,
            },
        }
    except Exception as exception:  # noqa: BLE001
        logger.error(
            "[Stacking] 추론 실패. error_type=%s",
            type(exception).__name__,
        )

        return {
            "engine": "stacking",
            "is_available": False,
            "result": {
                "risk_score": None,
                "risk_probability": None,
                "confidence": 0.0,
                "is_suspected_phishing": None,
                "threshold": None,
                "model_scores": {},
                "unavailable_models": [],
                "error_message": "Stacking Inference Error",
            },
        }

def reset_stacking_model_for_test() -> None:
    """테스트 간 전역 artifact 상태를 초기화"""

    global _load_attempted
    global _classifier
    global _load_error

    with _load_lock:
        _load_attempted = False
        _classifier = None
        _load_error = None