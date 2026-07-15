import pytest
from unittest.mock import patch
from app.service.security.parser import determine_final_threat_grade, analyze_threat_pipeline

def test_determine_final_threat_grade_safe():
    """
    위협 요소가 전혀 감지되지 않았을 때 SAFE 등급과 친근한 문구를 제대로 반환하는지 검증합니다.
    """
    url = "https://www.naver.com"
    gsb_malicious = False
    vt_stats = {"malicious": 0, "suspicious": 0, "harmless": 70, "undetected": 10}

    result = determine_final_threat_grade(url, gsb_malicious, vt_stats)

    assert result["result"]["grade"] == "SAFE"
    assert result["result"]["reason"] == "안심하고 접속하셔도 괜찮은 안전한 링크입니다."
    assert result["result"]["details"]["virustotal"]["malicious_engines"] == 0


def test_determine_final_threat_grade_suspicious():
    """
    바이러스토탈에서 1~2개의 엔진만 감지했거나 의심 요소가 있을 때 SUSPICIOUS 등급을 판정하는지 검증합니다.
    """
    url = "https://caution-link.com"
    gsb_malicious = False
    # malicious 카운트가 2개인 애매한 경우
    vt_stats = {"malicious": 2, "suspicious": 1, "harmless": 50, "undetected": 15}

    result = determine_final_threat_grade(url, gsb_malicious, vt_stats)

    assert result["result"]["grade"] == "SUSPICIOUS"
    assert result["result"]["reason"] == "잠재적 위협이 의심되는 페이지입니다. 접속 시 주의하시기 바랍니다."


def test_determine_final_threat_grade_dangerous_by_gsb():
    """
    Google Safe Browsing(1차 필터)에서 걸려 즉시 DANGEROUS 등급이 매겨지는지 검증합니다.
    """
    url = "https://danger-phishing-test-site.com"
    gsb_malicious = True
    vt_stats = {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0}

    result = determine_final_threat_grade(url, gsb_malicious, vt_stats)

    assert result["result"]["grade"] == "DANGEROUS"
    assert "안전하지 않은 사이트입니다. 악성코드 유포 혹은 피싱 사기 페이지로 감지되었습니다." in result["result"]["reason"]


def test_determine_final_threat_grade_dangerous_by_vt():
    """
    구글 세이프 브라우징은 뚫렸지만 바이러스토탈 백신 엔진 3개 이상에서 잡혀 DANGEROUS 판정이 나는지 검증합니다.
    """
    url = "https://hidden-malware-link.xyz"
    gsb_malicious = False
    # 백신 엔진 5개 감지
    vt_stats = {"malicious": 5, "suspicious": 2, "harmless": 40, "undetected": 20}

    result = determine_final_threat_grade(url, gsb_malicious, vt_stats)

    assert result["result"]["grade"] == "DANGEROUS"
    assert result["result"]["details"]["virustotal"]["malicious_engines"] == 5


@pytest.mark.asyncio
@patch("app.service.security.parser.is_mock_enabled", return_value=True)
async def test_analyze_threat_pipeline_mock_mode(mock_is_enabled):
    """
    Mock 스위치가 켜졌을 때 실제 API를 호출하는 대신 mock_provider의 사전에 미리 정의해 둔
    데이터베이스 결과로 우회 처리되어 빠른 응답이 성공적으로 가공되는지 비동기 통합 테스트를 수행합니다.
    """
    # mock_provider에 명시된 가상 안전 주소
    url = "https://www.google.com"
    
    result = await analyze_threat_pipeline(url)
    
    assert result["url"] == url
    assert result["is_mock"] is True
    assert result["result"]["grade"] == "SAFE"
    assert result["result"]["details"]["virustotal"]["harmless_engines"] == 72