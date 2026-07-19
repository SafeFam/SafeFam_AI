import os
import json
import logging
import httpx
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# 위험도 등급 산정 임계값 (TRAINING_FLOW.md의 SMS/Voice 모델 HIGH/MEDIUM/LOW 기준과 동일)
RISK_HIGH_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 40

MOCK_ENABLED = os.getenv("MOCK_SECURITY_API", "False").lower() in ("true", "1", "t")

# Gemini에게 구조화된 JSON 응답을 강제하기 위한 응답 스키마
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "risk_score": {
            "type": "INTEGER",
            "description": "문자 메시지의 스미싱 위험도 점수 (0~100)"
        },
        "tone_analysis": {
            "type": "STRING",
            "description": "메시지 어조 분석 (긴급성 유도, 공포/협박, 기관 사칭, 과도한 친밀감 등)"
        },
        "evidence": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "위험도 판단의 구체적 근거가 된 문장/표현 인용 목록"
        },
        "reason": {
            "type": "STRING",
            "description": "종합 판단 이유 요약"
        }
    },
    "required": ["risk_score", "tone_analysis", "evidence", "reason"]
}

SYSTEM_PROMPT = (
    "당신은 스미싱(문자 피싱) 탐지 전문가입니다. "
    "주어진 문자 메시지의 어조와 문맥을 분석하여 위험도를 판정하세요.\n\n"
    "분석 시 다음을 종합적으로 고려하세요:\n"
    "1. 어조(tone): 긴급성 유도, 공포/협박, 권위 사칭(수사기관·금융기관·택배사 등), 과도한 친밀감\n"
    "2. 근거(evidence): 메시지에서 실제로 의심스러운 표현이나 패턴을 구체적으로 인용\n\n"
    "반드시 지정된 JSON 스키마 형식으로만 응답하세요."
)

DEFAULT_ANALYSIS_RESULT = {
    "risk_score": 0,
    "tone_analysis": "분석 불가",
    "evidence": [],
    "reason": "Gemini API 호출에 실패하여 위험도를 판정할 수 없습니다."
}


def determine_text_risk_grade(risk_score: int) -> str:
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "DANGEROUS"
    elif risk_score >= RISK_MEDIUM_THRESHOLD:
        return "SUSPICIOUS"
    return "SAFE"


def _build_result(text_data: dict, is_mock: bool, error: str | None = None) -> dict:
    risk_score = text_data.get("risk_score", 0)

    # Fail-closed: 분석 실패 시 SAFE로 떨어지지 않도록 판정 불가 등급을 별도로 강제
    grade = "UNKNOWN" if error else determine_text_risk_grade(risk_score)

    result = {
        "is_mock": is_mock,
        "result": {
            "grade": grade,
            "risk_score": risk_score,
            "tone_analysis": text_data.get("tone_analysis", ""),
            "evidence": text_data.get("evidence", []),
            "reason": text_data.get("reason", "")
        }
    }

    if error:
        result["result"]["error_message"] = error

    return result


# Gemini API를 사용하여 문자 메시지의 어조/근거 기반 위험도 분석을 수행
async def analyze_text_with_gemini(text: str) -> dict:

    # mocking 체크 여부
    if MOCK_ENABLED:
        logger.info(f"[Mock Gemini] 실제 API 호출 우회 (Sandbox Mode)")
        mock_data = {
            "risk_score": 85,
            "tone_analysis": "긴급성 유도 및 기관 사칭 권위적 어조 감지",
            "evidence": ["발급 완료. 즉시 확인하세요.", "국민건강보험"],
            "reason": "[시연용 데이터] 건강검진 보고서 형식을 사칭하여 사용자의 급박한 클릭을 유도하는 전형적인 피싱 패턴입니다."
        }
        return _build_result(mock_data, is_mock=True)

    if not GEMINI_API_KEY:
        logger.warning("Gemini API Key가 누락되었습니다.")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Missing API Key")

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"parts": [{"text": f"분석할 문자 메시지:\n\"\"\"\n{text}\n\"\"\""}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.1
        }
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "content-type": "application/json"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, json=payload, headers=headers, timeout=10.0)

            if response.status_code == 429:
                logger.error("[Gemini] API 호출 한도 초과 (Rate Limit)")
                return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Rate Limit")

            response.raise_for_status()
            result_json = response.json()

            candidates = result_json.get("candidates", [])
            if not candidates:
                logger.error(f"[Gemini] 응답에 candidates가 없습니다: {result_json}")
                return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Empty Response")

            raw_text = candidates[0]["content"]["parts"][0]["text"]
            text_data = json.loads(raw_text)

            logger.info(f"[Gemini] 문자 분석 완료 - 위험도 점수: {text_data.get('risk_score', 0)}")
            return _build_result(text_data, is_mock=False)

    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API 에러 ({e.response.status_code}): {str(e)}")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="HTTP Error")

    except httpx.TimeoutException:
        logger.error("Gemini API 요청 타임아웃 발생")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Timeout")

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Gemini 응답 파싱 실패: {str(e)}")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Parse Error")

    except Exception as e:
        logger.error(f"Gemini 연동 중 비정상 에러 발생: {str(e)}")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Unknown Error")
