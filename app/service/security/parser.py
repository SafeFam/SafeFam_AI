import logging
from app.service.security.google_safe_browsing import check_google_safe_browsing
from app.service.security.virustotal import check_virus_total
from app.service.security.mock_provider import is_mock_enabled, get_mock_security_data

logger = logging.getLogger(__name__)

# Google Safe Browsing과 VirusTotal을 결합한 하이브리드 보안 검사 파이프라인
async def analyze_threat_pipeline(url: str) -> dict:

    # Mocking 여부 체크
    if is_mock_enabled():
        mock_data = get_mock_security_data(url)
        gsb_malicious = mock_data["google_safe_browsing"]
        vt_status = mock_data["virustotal"]

        return determine_final_threat_grade(url, gsb_malicious, vt_status, is_mock=True)

    # Google Safe Browsing 1차 스캔
    logger.info(f"[Pipeline] 1단계 - Google Safe Browsing 검증 시작: {url}")  
    gab_malicious = await check_google_safe_browsing(url)

    # 구글 블랙리스트에 등록되어 있으면 차단 처리
    if gsb_malicious:
        logger.warning(f"[Pipeline] 1단계 구글 필터 감지 차단 -> 즉시 DANGEROUS 반환: {url}")
        vt_stats = {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0}
        return determine_final_threat_grade(url, gsb_malicious, vt_stats)

    # VirusTotal 2차 종합 정밀 분석 (구글 필터 통과한 미지의 링크만 전송)
    logger.info(f"[Pipeline] 2단계 - VirusTotal 정밀 스캔 시작: {url}")
    vt_stats = await check_virus_total(url)

    return determine_final_threat_grade(url, gsb_malicious, vt_stats)

# Google Safe browsing 결과와 VirusTotal 탐지 통계를 바탕으로,
# 최종 위험도 등급(SAFE, SUSPICIOUS, DANGEROUS) 및 판정 이유를 산출
def determine_final_threat_grade(url: str, gsb_malicious: bool, vt_stats: dict, is_mock: bool = False) -> dict:
    malicious_count = vt_status.get("malicious", 0)
    suspicious_count = vt_status.get("suspicious", 0)

    # 기본값 안전(SAFE) 설정
    grade = "SAFE"
    reason = "안심하고 접속하셔도 괜찮은 안전한 링크입니다.."

    # 구글 필터링에서 감지되었거나, 바이러스토탈 백신 엔진 중 3개 이상이 악성으로 분류한 경우
    if gsb_malicious or malicious_count >= 3:
        grade = "DANGEROUS"
        reason = "안전하지 않은 사이트입니다. 악성코드 유포 혹은 피싱 사기 페이지로 감지되었습니다."

    # 바이러스토탈에서 1~2개 엔진에 의해 악성으로 감지되었거나 의심 카운트가 있는 경우
    elif 0 < malicious_count < 3 or suspicious_count > 0:
        grade = "SUSPICIOUS"
        reason = "잠재적 위협이 의심되는 페이지입니다. 접속 시 주의하시기 바랍니다."

    result = {
        "url": url,
        "is_mock": is_mock,
        "result": {
            "grade": grade,
            "reason": reason,
            "details": {
                "google_safe_browsing": {
                    "detected": gsb_malicious
                },
                "virustotal": {
                    "malicious_engines": malicious_count,
                    "suspicious_engines": suspicious_count,
                    "harmless_engines": vt_stats.get("harmless", 0),
                    "total_analyzed_engines": sum(vt_stats.values()) - (1 if "error" in vt_stats or "status" in vt_stats else 0)
                }
            }
        }
    }

    # 만약 VirusTotal Rate Limit 등으로 인한 에러 반환 기록이 있다면 응답에 패키징
    if "error" in vt_stats:
        result["result"]["details"]["virustotal"]["error_message"] = vt_stats["error"]

    logger.info(f"[Pipeline Final] {url} 분석 완료 -> Grade: {grade}")
    return result