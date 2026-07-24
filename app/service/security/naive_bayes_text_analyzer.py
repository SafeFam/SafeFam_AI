import os
import re
import logging
from pathlib import Path

from app.service.security.gemini_text_analyzer import determine_text_risk_grade

logger = logging.getLogger(__name__)

# 프로젝트 루트 기준 사전 학습된 아티팩트 위치 (data_science/SMSModel/train_sms.py 산출물)
_BASE_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_DIR = _BASE_DIR / "data_science" / "SMSModel"

MODEL_PATH = Path(os.getenv("NAIVE_BAYES_MODEL_PATH", str(_DEFAULT_MODEL_DIR / "phishing_model_artifact.pkl")))
VECTORIZER_PATH = Path(os.getenv("NAIVE_BAYES_VECTORIZER_PATH", str(_DEFAULT_MODEL_DIR / "phishing_vectorizer.pkl")))

# --- 전처리 정규식 : data_science/SMSModel/train_sms.py의 정규화/피처 추출 로직과 반드시 동일하게 유지 ---
# (학습 시 벡터라이저가 본 입력 분포와 서빙 시 입력 분포가 어긋나면 모델이 무의미해짐)
_RE_URL = re.compile(r"https?://\S+|[a-zA-Z0-9.-]+\.(kr|com|net|cyou|xyz|me|io|cc)\S*")
_RE_PHONE = re.compile(r"\d{2,4}-\d{3,4}-\d{4}")
_RE_LONG_NUM = re.compile(r"\b\d{6,}\b")
_RE_AMOUNT = re.compile(r"\d+[,\d]*원")
_RE_FORMAT_ARTIFACT = re.compile(r"={2,}|■|□|▪|▫|●|○|\s-\s|\s:\s")
_RE_SHORT_URL = re.compile(r"bit\.ly|goo\.gl|tinyurl|gourl|ow\.ly|n\.bnuee|han\.gl|cutt\.ly")
_RE_WEB_TAG = re.compile(r"\[Web발신\]|\[국외발신\]|\[국제발신\]")

DEFAULT_ANALYSIS_RESULT = {
    "grade": "UNKNOWN",
    "risk_score": 0,
    "is_suspected_phishing": False,
    "error_message": "나이브 베이즈 모델을 로드하지 못해 위험도를 판정할 수 없습니다."
}

_model = None
_vectorizer = None
_threshold = None
_classes = None
_load_error: str | None = None
_load_attempted = False


def _normalize_text(text: str) -> str:
    text = _RE_URL.sub("<URL>", text)
    text = _RE_PHONE.sub("<전화번호>", text)
    text = _RE_LONG_NUM.sub("<긴숫자>", text)
    text = _RE_AMOUNT.sub("<금액>", text)
    text = _RE_FORMAT_ARTIFACT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_struct_features(text: str) -> list:
    return [
        int(bool(_RE_URL.search(text))),
        int(bool(_RE_SHORT_URL.search(text))),
        int(bool(_RE_PHONE.search(text))),
        int(bool(_RE_AMOUNT.search(text))),
        int(bool(_RE_WEB_TAG.search(text))),
        int(len(text) > 100),
    ]


def _load_artifacts() -> None:
    """FastAPI 프로세스 당 1회만 시도. 실패 시 재시도하지 않고 fail-safe 응답으로 대체."""
    global _model, _vectorizer, _threshold, _classes, _load_error, _load_attempted

    if _load_attempted:
        return
    _load_attempted = True

    try:
        import joblib

        artifact = joblib.load(MODEL_PATH)
        _model = artifact["model"]
        _threshold = artifact["threshold"]
        _classes = artifact["classes"]
        _vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info(f"[NaiveBayes] 모델 로드 완료 (threshold={_threshold})")
    except Exception as e:
        _load_error = str(e)
        logger.error(f"[NaiveBayes] 모델 로드 실패: {_load_error}")


def is_model_loaded() -> bool:
    _load_artifacts()
    return _model is not None


# 사전 학습된 나이브 베이즈(ComplementNB + isotonic 보정) 모델로 문자 메시지의 1차 위험도를 산출
async def analyze_text_with_naive_bayes(text: str) -> dict:
    _load_artifacts()

    if _model is None:
        return {
            "engine": "naive_bayes",
            "is_available": False,
            "result": dict(DEFAULT_ANALYSIS_RESULT, error_message=_load_error or DEFAULT_ANALYSIS_RESULT["error_message"])
        }

    try:
        import numpy as np
        from scipy.sparse import csr_matrix, hstack

        text_norm = _normalize_text(text)
        struct = np.array([_extract_struct_features(text)])

        X_text = _vectorizer.transform([text_norm])
        X = hstack([X_text, csr_matrix(struct)])

        phishing_idx = _classes.index("phishing")
        prob_phishing = _model.predict_proba(X)[0][phishing_idx]
        risk_score = int(prob_phishing * 100)

        logger.info(f"[NaiveBayes] 문자 분석 완료 - 위험도 점수: {risk_score}")

        return {
            "engine": "naive_bayes",
            "is_available": True,
            "result": {
                "grade": determine_text_risk_grade(risk_score),
                "risk_score": risk_score,
                "is_suspected_phishing": bool(prob_phishing >= _threshold),
                "error_message": None
            }
        }
    except Exception as e:
        logger.error(f"[NaiveBayes] 추론 중 비정상 에러 발생: {str(e)}")
        return {
            "engine": "naive_bayes",
            "is_available": False,
            "result": dict(DEFAULT_ANALYSIS_RESULT, error_message="Inference Error")
        }
