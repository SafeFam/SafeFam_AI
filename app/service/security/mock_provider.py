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