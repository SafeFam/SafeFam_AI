import os 
import logging 
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 스위치 flag 로드(기본값은 실제 호출하도록 False)
MOCK_ENABLED = os.getenv("MOCK_SECURITY_API", "False").lower() in ("true", "1", "yes")

# 시연 및 테스트용 사전 정의 데이터베이스 (Mock DB)
MOCK_DATABASE = {
    "https://www.google.com": {
        "google_safe_browsing": False,  # 안전
        "virustotal": {
            "malicious": 0,
            "suspicious": 0,
            "harmless": 72,
            "undetected": 10
        }
    },
    "https://www.naver.com": {
        "google_safe_browsing": False,  # 안전
        "virustotal": {
            "malicious": 0,
            "suspicious": 0,
            "harmless": 75,
            "undetected": 5
        }
    },
    # 시연용 악성 피싱 사이트 가상 URL 예시
    "https://danger-phishing-test-site.com": {
        "google_safe_browsing": True,   # 구글 필터 감지
        "virustotal": {
            "malicious": 14,
            "suspicious": 3,
            "harmless": 40,
            "undetected": 15
        }
    },
    "https://hidden-malware-link.xyz": {
        "google_safe_browsing": False,  # 구글은 뚫렸지만
        "virustotal": {
            "malicious": 8,            # 바이러스토탈에서 잡힌 케이스
            "suspicious": 2,
            "harmless": 50,
            "undetected": 12
        }
    }
}

# 기본 모크 응답 (DB에 정의되지 않은 URL 입력 시 반환하는 기본 데이터)
DEFAULT_SAFE_MOCK = {
    "google_safe_browsing": False,
    "virustotal": {
        "malicious": 0,
        "suspicious": 0,
        "harmless": 65,
        "undetected": 15
    }
}

def is_mock_enabled() -> bool:
    return MOCK_ENABLED

def get_mock_security_data(url: str) -> dict:
    logger.info(f"[Mock Security API] 실제 API 호출 우회 (Sandbox Mode) -> {url}")
    return MOCK_DATABASE.get(url, DEFAULT_SAFE_MOCK)

# 시연 및 테스트용 사전 정의 문자 위험도 분석 결과 (Gemini Mock DB)
MOCK_TEXT_DATABASE = {
    "[Web발신] 안녕하세요 고객님, 주문하신 상품이 배송 완료되었습니다.": {
        "risk_score": 3,
        "tone_analysis": "일반적인 배송 안내 문구로 긴급성 유도나 협박 어조가 없습니다.",
        "evidence": [],
        "reason": "정상적인 배송 완료 안내 메시지로 판단됩니다."
    },
    # 시연용 악성 스미싱 텍스트 예시
    "[검찰청] 귀하 명의로 대포통장이 개설되어 수사가 진행 중입니다. 즉시 아래 링크로 접속하여 신원을 확인하세요.": {
        "risk_score": 96,
        "tone_analysis": "수사기관을 사칭하며 즉각적인 공포와 긴급성을 유도하는 전형적인 협박성 어조입니다.",
        "evidence": [
            "'검찰청' 명의로 발신자를 사칭",
            "'대포통장', '수사가 진행 중' 등 공포 유발 표현 사용",
            "'즉시', '접속하여 확인' 등 즉각적 행동을 강요하는 표현"
        ],
        "reason": "기관 사칭과 협박성 문구, 즉각적인 링크 접속 유도가 결합된 전형적인 스미싱 패턴입니다."
    }
}

# 기본 모크 응답 (DB에 정의되지 않은 텍스트 입력 시 반환하는 기본 데이터)
DEFAULT_SAFE_TEXT_MOCK = {
    "risk_score": 5,
    "tone_analysis": "특이 어조가 감지되지 않았습니다.",
    "evidence": [],
    "reason": "위험 신호가 발견되지 않은 일반적인 메시지입니다."
}

def get_mock_text_analysis_data(text: str) -> dict:
    logger.info("[Mock Gemini] 실제 API 호출 우회 (Sandbox Mode)")
    return MOCK_TEXT_DATABASE.get(text, DEFAULT_SAFE_TEXT_MOCK)
