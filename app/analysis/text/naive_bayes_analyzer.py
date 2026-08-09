# 사전 학습된 Naive Bayes 모델을 이용한 SMS 피싱 분석기
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from app.analysis.risk_policy import determine_text_risk_grade
from app.analysis.text.preprocessing import (
    extract_struct_features,
    normalize_text,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


MODEL_PATH = settings.NAIVE_BAYES_MODEL_PATH
VECTORIZER_PATH = settings.NAIVE_BAYES_VECTORIZER_PATH


DEFAULT_ANALYSIS_RESULT = {
    "grade": "UNKNOWN",
    "risk_score": 0,
    "is_suspected_phishing": False,
    "error_message": (
        "나이브 베이즈 모델을 로드하지 못해 위험도를 판정할 수 없습니다."
    ),
}

_model = None
_vectorizer = None
_threshold = None
_classes = None

_load_error: str | None = None
_load_attempted = False
_load_lock = threading.Lock()


def resolve_artifact_paths(
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
) -> tuple[Path, Path]:
    """공유 포인터가 있으면 같은 세대의 모델과 벡터라이저 경로를 반환"""
    model_path = Path(MODEL_PATH if model_path is None else model_path)
    vectorizer_path = Path(
        VECTORIZER_PATH if vectorizer_path is None else vectorizer_path
    )
    pointer_path = model_path.parent / "current.json"
    if not pointer_path.is_file():
        return model_path, vectorizer_path

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    version_dir = model_path.parent / "versions" / pointer["generation"]
    return version_dir / pointer["model"], version_dir / pointer["vectorizer"]


def _load_artifacts() -> None:
    """
    모델과 벡터라이저를 프로세스당 한 번만 로드

    로드에 실패해도 API 서버 전체를 중단시키지 않고, 이후 분석 요청에서
    UNKNOWN 결과를 반환할 수 있도록 오류 상태만 저장
    """
    global _model
    global _vectorizer
    global _threshold
    global _classes
    global _load_error
    global _load_attempted

    # 1차 체크
    if _load_attempted:
        return

    # 스레드 락 적용
    with _load_lock:
        # 2차 체크
        if _load_attempted:
            return

        try:
            import joblib

            model_path, vectorizer_path = resolve_artifact_paths()
            artifact = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

            # 모든 값이 정상적으로 읽힌 이후 전역 상태를 갱신
            _model = artifact["model"]
            _threshold = artifact["threshold"]
            _classes = artifact["classes"]
            _vectorizer = vectorizer
            _load_error = None

            logger.info(
                "[NaiveBayes] 모델 로드 완료 (threshold=%s)",
                _threshold,
            )
        except Exception as exception:  # noqa: BLE001 - artifact 오류는 fail-safe 처리
            _load_error = type(exception).__name__

            logger.error(
                "[NaiveBayes] 모델 로드 실패. error_type=%s",
                _load_error,
            )
        finally:
            _load_attempted = True


def is_model_loaded() -> bool:
    """Naive Bayes 모델을 사용할 수 있는지 반환"""
    _load_artifacts()
    return _model is not None


async def analyze_text_with_naive_bayes(text: str) -> dict:
    """사전 학습된 Naive Bayes 모델로 SMS의 피싱 위험도를 분석"""
    _load_artifacts()

    if _model is None:
        return {
            "engine": "naive_bayes",
            "is_available": False,
            "result": dict(DEFAULT_ANALYSIS_RESULT),
        }

    try:
        # 학습과 동일한 공통 전처리를 적용
        normalized_text = normalize_text(text)

        struct_features = np.asarray(
            [extract_struct_features(text)],
            dtype=np.int8,
        )

        # 기존 artifact가 기대하는 입력 구조를 유지
        text_features = _vectorizer.transform([normalized_text])
        feature_matrix = hstack(
            [
                text_features,
                csr_matrix(struct_features),
            ]
        )

        phishing_index = _classes.index("phishing")
        phishing_probability = _model.predict_proba(feature_matrix)[0][phishing_index]

        risk_score = int(phishing_probability * 100)

        logger.info(
            "[NaiveBayes] 문자 분석 완료 - 위험도 점수: %s",
            risk_score,
        )

        return {
            "engine": "naive_bayes",
            "is_available": True,
            "result": {
                "grade": determine_text_risk_grade(risk_score),
                "risk_score": risk_score,
                "is_suspected_phishing": bool(phishing_probability >= _threshold),
                "error_message": None,
            },
        }
    except Exception as exception:  # noqa: BLE001 - 추론 오류는 fail-safe 처리
        logger.error(
            "[NaiveBayes] 추론 중 비정상 오류 발생. error_type=%s",
            type(exception).__name__,
        )

        return {
            "engine": "naive_bayes",
            "is_available": False,
            "result": dict(
                DEFAULT_ANALYSIS_RESULT,
                error_message="Inference Error",
            ),
        }
