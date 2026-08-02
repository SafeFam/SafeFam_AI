import os 
import logging 

logger = logging.getLogger(__name__)

# [1] URL 보안 스캔용 Mock 엔진 (클래스 구조화)
class MockSecurityProvider:
    def __init__(self):
        self.mock_db = {
            "https://www.google.com": {
                "google_safe_browsing": False,
                "malicious_count": 0
            },
            "https://www.naver.com": {
                "google_safe_browsing": False,
                "malicious_count": 0
            },
            "https://danger-phishing-test-site.com": {
                "google_safe_browsing": True,
                "malicious_count": 14
            },
            "https://hidden-malware-link.xyz": {
                "google_safe_browsing": False,
                "malicious_count": 8
            }
        }
        self.default_safe = {"google_safe_browsing": False, "malicious_count": 0}

    async def scan_url(self, url: str) -> dict:
        """
        [상용 인터페이스 구현]
        가상 DB를 매칭하여 상용 엔진과 동일한 규격의 딕셔너리를 반환합니다.
        """
        logger.info("[MOCK SECURITY ENGINE] Sandbox API 우회 매칭")
        
        # URL이 DB에 없으면 안전한 상태인 default_safe 적용
        data = self.mock_db.get(url, self.default_safe)
        
        gsb_malicious = data.get("google_safe_browsing", False)
        vt_malicious_count = data.get("malicious_count", 0)
        
        # 악성 여부 및 위험도 점수 가중치 산정 (상용 규격과 통일)
        is_malicious = gsb_malicious or (vt_malicious_count >= 3)
        
        if gsb_malicious:
            raw_score = 0.95
        elif vt_malicious_count > 0:
            raw_score = min(0.2 + (vt_malicious_count * 0.08), 1.0)
        else:
            raw_score = 0.0

        return {
            "is_malicious": is_malicious,
            "raw_score": round(raw_score, 2),
            "detected_count": vt_malicious_count,
            "status": "completed"
        }


# [2] 텍스트 위험도 분석용 Mock 데이터 (기존 자산 완벽 보존)
MOCK_TEXT_DATABASE = {
    "[Web발신] 안녕하세요 고객님, 주문하신 상품이 배송 완료되었습니다.": {
        "risk_score": 3,
        "tone_analysis": "일반적인 배송 안내 문구로 긴급성 유도나 협박 어조가 없습니다.",
        "evidence": [],
        "reason": "정상적인 배송 완료 안내 메시지로 판단됩니다."
    },
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

DEFAULT_SAFE_TEXT_MOCK = {
    "risk_score": 5,
    "tone_analysis": "특이 어조가 감지되지 않았습니다.",
    "evidence": [],
    "reason": "위험 신호가 발견되지 않은 일반적인 메시지입니다."
}

def get_mock_text_analysis_data(text: str) -> dict:
    """
    [기존 자산 보존] 추후 Gemini AI 모델 API 연동 고도화 단계에서 활용할 텍스트 샌드박스 데이터 함수
    """
    logger.info("[Mock Gemini] 실제 AI API 호출 우회 (Sandbox Mode)")
    return MOCK_TEXT_DATABASE.get(text, DEFAULT_SAFE_TEXT_MOCK)
