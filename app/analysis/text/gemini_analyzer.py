import json
import logging

import httpx

from app.analysis.risk_policy import determine_text_risk_grade
from app.core.config import settings
from app.infrastructure.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MOCK_ENABLED = settings.MOCK_SECURITY_API

# Gemini에게 구조화된 JSON 응답을 강제하기 위한 응답 스키마
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "risk_score": {
            "type": "INTEGER",
            "description": "문자 메시지의 스미싱 위험도 점수 (0~100 사이의 정수. SAFE인 일상 문장은 0~20 사이로 측정)",
        },
        "tone_analysis": {
            "type": "STRING",
            "description": "메시지 어조 분석 (예: '평범한 가족 간의 일상 대화', '긴급성 유도 및 기관 사칭' 등)",
        },
        "evidence": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "위험도 판단 근거를 AI가 분석하여 생성한 설명 목록 (원문 인용 금지. SAFE인 경우 반드시 빈 리스트 [])",
        },
        "reason": {"type": "STRING", "description": "종합 판단 이유 요약"},
    },
    "required": ["risk_score", "tone_analysis", "evidence", "reason"],
}

SYSTEM_PROMPT = (
    "You are an advanced, production-grade 'AI Smishing Detection Engine' deployed in a financial security infrastructure. "
    "Your core objective is to audit incoming SMS/MMS messages, perform semantic context analysis, and isolate malicious intent "
    "while rigorously protecting benign personal conversations from False Positives (오탐).\n\n"
    "--- CORE TRIAGE RULE (오탐 원천 차단 절대 규칙) ---\n"
    "- Personal, trivial, or informal interactions between family, friends, or colleagues (e.g., '엄마 오늘 저녁 메뉴 뭐야?', '나 늦을 거 같아', '오늘 소주 한잔 고?') "
    "MUST be unconditionally classified with a risk_score between 0 and 15, and isolated as 'SAFE'.\n"
    "- Unless there is an explicit presence of institutional impersonation (사칭), urgent coercive threats (협박), identity fraud, social engineering extortion, "
    "or unverified sideloading file paths, DO NOT escalate the severity to SUSPICIOUS or DANGEROUS.\n"
    "- Polite or formal tone is NEVER a safety signal on its own (정중하거나 격식 있는 문체 자체는 안전 신호가 아니다). Formal greetings ('안녕하십니까'), "
    "gratitude phrases ('이용해 주셔서 감사합니다'), bulleted official-looking layouts ('■ 항목: 값'), and polite closings ('감사합니다') are increasingly used "
    "specifically to mimic authentic corporate/institutional notices and evade tone-based detection. If impersonation, account/institution details, or a redirect "
    "to an unverified 'customer center'/'report line' co-occur with this formal tone, DO NOT lower the risk_score because the message 'sounds official' — "
    "evaluate purely on the presence of indicators 1-5 below.\n\n"
    "--- SEMANTIC AUDIT MATRIX (위험도 산정 지표) ---\n"
    "Evaluate the text against the following smishing indicators:\n"
    "1. Impersonation (사칭): Posing as financial institutions, public services, judicial authorities, or courier services (e.g., 국민건강보험, 법원, 택배사, 시중은행).\n"
    "2. Urgency & Coercion (긴급성/압박): Forcing an immediate behavioral action via fear-inducing or urgent deadlines (e.g., '즉시 확인 요망', '계좌 정지 예정', '과태료 처분').\n"
    "3. Social Engineering Baiting (사회공학적 유도): Fabricating plausible crises or benefits to trigger high emotional distress or greed.\n"
    "4. Sideloading/Malware Triggers (악성 유도): Forcing or enticing credentials, credential updates, personal identification disclosure, or third-party interactions.\n"
    "5. Tone Camouflage (문체 위장): Imitating authentic corporate/institutional notice formatting (formal greetings, bulleted transaction/account details, polite "
    "closings) while still directing the recipient to an unverified contact channel for 'identity theft' or 'unauthorized transaction' verification. This is a "
    "known evasion tactic that masks indicators 1-4 behind a calm, official-sounding surface — treat it as an aggravating signal, not a mitigating one.\n\n"
    "--- FEW-SHOT AUDIT REFERENCE (분석 참조 예시) ---\n"
    "Example 1 (Benign / Casual text):\n"
    "  Input: '엄마 오늘 저녁 메뉴 뭐야?'\n"
    "  Output: {\n"
    '    "risk_score": 0,\n'
    '    "tone_analysis": "지인 간의 지극히 평범하고 일상적인 대화 어조",\n'
    '    "evidence": [],\n'
    '    "reason": "어떠한 사회공학적 유도 기법, 기관 사칭, 또는 금전 요구나 정보 유출 시도가 포함되지 않은 단순 가족 간의 일상 메시지이므로 완벽히 안전합니다."\n'
    "  }\n\n"
    "Example 2 (Malicious Smishing text):\n"
    "  Input: '[국민건강보험] 건강검진 보고서 발급 완료. 즉시 확인하세요 http://bit.ly/fake'\n"
    "  Output: {\n"
    '    "risk_score": 90,\n'
    '    "tone_analysis": "공공기관 사칭 및 데드라인 설정을 통한 심리적 긴급성 유도 어조",\n'
    '    "evidence": ["공공기관 사칭이 감지되었습니다.", "긴급성을 유도하는 표현이 포함되어 있습니다."],\n'
    '    "reason": "공공기관인 국민건강보험공단을 사칭하고 있으며, \'즉시\'라는 표현으로 사용자의 불안감과 급박한 심리를 자극하여 첨부된 출처 불명의 악성 URL 링크 클릭을 유도하는 전형적인 기관 사칭형 피싱 메시지입니다."\n'
    "  }\n\n"
    "Example 3 (Malicious Smishing text disguised as a formal bank notice — Tone Camouflage):\n"
    "  Input: '[KB국민은행] 결제 완료 안내\\n\\n000 고객님, 요청하신 물품 구매 건에 대해 결제가 정상적으로 완료되었습니다.\\n\\n"
    "본인이 결제하지 않으셨거나 명의 도용 등의 금융 사고가 의심되는 경우, 즉시 아래의 고객센터로 연락하여 조치를 취하시기 바랍니다.\\n\\n"
    "■ 신고 및 문의처: 02-330-4284'\n"
    "  Output: {\n"
    '    "risk_score": 85,\n'
    '    "tone_analysis": "정중하고 격식 있는 은행 공지문 형식으로 위장했으나, 명의도용을 언급하며 미검증 연락처로 유도하는 전형적인 사칭형 문자",\n'
    '    "evidence": ["금융기관(KB국민은행) 사칭이 감지되었습니다.", "명의 도용을 언급하며 별도 연락처로의 확인을 유도합니다."],\n'
    '    "reason": "정중한 어조와 공식 공지문 형식(인사말, 항목별 안내, 정중한 마무리)을 사용했지만, 이는 사칭 사실을 감추기 위한 문체 위장(Tone Camouflage)일 뿐입니다. '
    "금융기관을 사칭하며 명의도용 가능성을 언급해 불안을 조성하고, 검증되지 않은 자체 연락처로 확인을 유도하는 구조는 문체와 무관하게 전형적인 스미싱 패턴이므로 격식체라는 이유로 위험도를 낮추지 않았습니다.\"\n"
    "  }\n\n"
    "--- OUTPUT COMPLIANCE ---\n"
    "- All text outputs (tone_analysis, reason) must be cleanly generated in Korean (한국어) for target enterprise consumption.\n"
    "- You must strictly adhere to the designated JSON schema format. Do not prepend markdown formatting inside the json payload."
)

DEFAULT_ANALYSIS_RESULT = {
    "risk_score": 0,
    "tone_analysis": "분석 불가",
    "evidence": [],
    "reason": "Gemini API 호출에 실패하여 위험도를 판정할 수 없습니다.",
}


def _build_result(text_data: dict, is_mock: bool, error: str | None = None) -> dict:
    risk_score = text_data.get("risk_score", 0)

    grade = "UNKNOWN" if error else determine_text_risk_grade(risk_score)

    result = {
        "is_mock": is_mock,
        "result": {
            "grade": grade,
            "risk_score": risk_score,
            "tone_analysis": text_data.get("tone_analysis", ""),
            "evidence": text_data.get("evidence", []),
            "reason": text_data.get("reason", ""),
        },
    }

    if error:
        result["result"]["error_message"] = error

    return result


# Gemini API를 사용하여 문자 메시지의 어조/근거 기반 위험도 분석을 수행
async def analyze_text_with_gemini(text: str) -> dict:

    # mocking 체크 여부
    if MOCK_ENABLED:
        logger.info("[Mock Gemini] 실제 API 호출 우회 (Sandbox Mode)")
        mock_data = {
            "risk_score": 85,
            "tone_analysis": "긴급성 유도 및 기관 사칭 권위적 어조 감지",
            "evidence": ["공공기관 사칭이 감지되었습니다.", "긴급성을 유도하는 표현이 포함되어 있습니다."],
            "reason": "[시연용 데이터] 건강검진 보고서 형식을 사칭하여 사용자의 급박한 클릭을 유도하는 전형적인 피싱 패턴입니다.",
        }
        return _build_result(mock_data, is_mock=True)

    if not GEMINI_API_KEY:
        logger.warning("Gemini API Key가 누락되었습니다.")
        return _build_result(
            DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Missing API Key"
        )

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": f'분석할 문자 메시지:\n"""\n{text}\n"""'}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "temperature": 0.1,
        },
    }

    try:
        result_json = await GeminiClient().generate(
            api_url=API_URL,
            api_key=GEMINI_API_KEY,
            payload=payload,
        )

        candidates = result_json.get("candidates", [])
        if not candidates:
            logger.error("[Gemini] 응답에 candidates가 없습니다.")
            return _build_result(
                DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Empty Response"
            )

        raw_text = candidates[0]["content"]["parts"][0]["text"]
        text_data = json.loads(raw_text)

        logger.info(
            f"[Gemini] 문자 분석 완료 - 위험도 점수: {text_data.get('risk_score', 0)}"
        )
        return _build_result(text_data, is_mock=False)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.error("[Gemini] API 호출 한도 초과 (Rate Limit)")
            return _build_result(
                DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Rate Limit"
            )
        logger.error(f"Gemini API 에러 ({e.response.status_code})")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="HTTP Error")

    except httpx.TimeoutException:
        logger.error("Gemini API 요청 타임아웃 발생")
        return _build_result(DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Timeout")

    except (KeyError, IndexError, json.JSONDecodeError) as exception:
        logger.error(
            "Gemini 응답 파싱 실패. error_type=%s",
            type(exception).__name__,
        )
        return _build_result(
            DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Parse Error"
        )

    except Exception as exception:
        logger.error(
            "Gemini 연동 중 비정상 오류 발생. error_type=%s",
            type(exception).__name__,
        )
        return _build_result(
            DEFAULT_ANALYSIS_RESULT, is_mock=False, error="Unknown Error"
        )
