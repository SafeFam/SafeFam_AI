import os
import logging
import base64
import httpx
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv

logger = logging.getLogger(__name__)

VT_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
VT_BASE_URL= "https://www.virustotal.com/api/v3"

# VirusTotal v3 API 규격에 맞게 URL을 Base64로 인코딩하여 URL ID를 생성
def get_url_id(url: str) -> str:

    # URL을 UTF-8 바이트로 변환 후 base64 인코딩
    b64_bytes = base64.urlsafe_b64encode(url.encode("utf-8"))

    # '=' 패딩 문자를 제거한 문자열 반환
    return b64_bytes.decode("utf-8").rstrip("=")

# VirusTotal API를 사용하여 URL의 정밀 분석 결과를 가져오고, 
# 기존에 스캔된 이력이 없다면 신규 스캔 요청
async def check_virus_total(url: str) -> dict:

    if not VT_API_KEY:
        logger.warning("VirusTotal API Key가 누락되었습니다. 빈 분석 결과를 반환합니다.")
        return {"malicious": 0, "suspicious": 0, "harmless": 0. "undetected": 0}

    url_id = get_url_id(url)
    headers = {
        "x-apikey": VT_API_KEY,
        "accept": "application/json"
    }    

    async with httpx.AsyncClient() as client:
        try:
            # 기존 분석 보고서 조회 시도 (GET)
            report_url = f"{VT_BASE_URL}/urls/{url_id}"
            response = await client.get(report_url, headers=headers, timeout=5.0)

            # 기존 보고서가 없을 때 신규 스캔 요청 (POST)
            if response.status_code == 404:
                logger.info(f"[VirusTotal] 기존 보고서 없음. 신규 스캔 요청 시작: {url}")
                scan_url = f"{VT_BASE_URL}/urls"
                scan_response = await client.post(scan_url, headers=headers, data={"url": url}, timeout=5.0)

                # 호출 한도 초과 등 에러 처리
                if scan_response.status_code == 429:
                    logger.error("[VirusTotal] API 호출 한도 초과 (Rate Limit)")
                    return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "error": "Rate Limit"}

                scan_response.raise_for_status()
                scan_result = scan_response.json()

                # 최초 요청 직후에는 기본 구조 데이터만 반환
                logger.info(f"[VirusTotal] 신규 스캔 요청 완료: {url}")
                return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "status": "scanning"}

            # API 한도 초과 에러 처리
            if response.status_code == 429:
                logger.error("[VirusTotal] API 호출 한도 초과 (Rate Limit)")
                return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "error": "Rate Limit"}

            response.raise_for_status()
            report_data = response.json()

            # 보고서 데이터 파싱 (마지막 분석 결과 추출)
            status = report_data.get("data", {}).get("attributes", {}).get("last_analysis_status", {})

            logger.info(f"[VirusTotal] 분석 완료 - 악성: {status.get('malicious', 0)}, 의심: {status.get('suspicious', 0)}")
            return {
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0)
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"VirusTotal API 에러 ({e.response.status_code}): {str(e)}")
            return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "error": "HTTP Error"}
        except httpx.TimeoutException:
            logger.error("VirusTotal API 요청 타임아웃 발생")
            return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "error": "Timeout"}
        except Exception as e:
            logger.error(f"VirusTotal 연동 중 비정상 에러 발생: {str(e)}")
            return {"malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0, "error": "Unknown Error"}