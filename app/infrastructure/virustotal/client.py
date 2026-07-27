import logging
import base64
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

class VirusTotalClient:
    def __init__(self):
        self.api_key = settings.VIRUSTOTAL_API_KEY
        self.base_url = "https://www.virustotal.com/api/v3"
        self.headers = {
            "x-apikey": self.api_key if self.api_key else "",
            "accept": "application/json"
        }

    # URL을 VirusTotal v3 규격에 맞게 Base64 URL-Safe 인코딩
    def _get_url_id(self, url: str) -> str:

        b64_bytes = base64.urlsafe_b64encode(url.encode("utf-8"))
        return b64_bytes.decode("utf-8").rstrip("=")

    # VirusTotal API를 호출해 악성 여부 및 상세 스코어 판정
    async def scan_url(self, url: str) -> dict:
        
        # 기본 안전 상태 스켈레톤 리턴 규격 정의
        default_result = {"is_malicious": False, "raw_score": 0.0, "detected_count": 0, "status": "safe"}
        
        if not self.api_key:
            logger.warning("[VirusTotal] API Key가 누락되었습니다. 빈 분석 결과를 반환합니다.")
            return default_result

        url_id = self._get_url_id(url)
        report_url = f"{self.base_url}/urls/{url_id}"

        async with httpx.AsyncClient() as client:
            try:
                # 기존 분석 보고서 조회 시도 
                response = await client.get(report_url, headers=self.headers, timeout=5.0)

                # 기존 보고서가 없을 때 신규 스캔 요청 분기 
                if response.status_code == 404:
                    logger.info(f"[VirusTotal] 기존 보고서 없음. 신규 스캔 요청 시작: {url}")
                    scan_url = f"{self.base_url}/urls"
                    scan_response = await client.post(scan_url, headers=self.headers, data={"url": url}, timeout=5.0)

                    if scan_response.status_code == 429:
                        logger.error("[VirusTotal] API 호출 한도 초과 (Rate Limit)")
                        return default_result

                    scan_response.raise_for_status()
                    logger.info(f"[VirusTotal] 신규 스캔 요청 완료: {url}")
                    return {"is_malicious": False, "raw_score": 0.0, "detected_count": 0, "status": "scanning"}

                # Rate Limit 에러 처리
                if response.status_code == 429:
                    logger.error("[VirusTotal] API 호출 한도 초과 (Rate Limit)")
                    return default_result

                response.raise_for_status()
                report_data = response.json()

                # 보고서 데이터 파싱 (오타 수정 및 디테일 파싱)
                stats = report_data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})

                malicious = stats.get("malicious", 0)
                suspicious = stats.get("suspicious", 0)
                # last_analysis_stats에 잡힌 전체 엔진 수 (malicious/suspicious/harmless/undetected/timeout 등 전부 합산)
                total_engines = sum(stats.values()) if stats else 0

                logger.info(
                    f"[VirusTotal] 분석 완료 - 악성: {malicious}, 의심: {suspicious}, 전체 엔진: {total_engines}"
                )

                # 백신 엔진 중 3개 이상이 악성(malicious)이라고 판정하거나, 의심 엔진이 과도하게 많을 때 악성으로 분류
                is_malicious = (malicious >= 3) or (malicious + suspicious >= 5)

                # 악성 판정 엔진 수 비율 기반 위험도 점수 산정 (의심 엔진은 절반 가중치로 반영, 최대 1.0)
                if total_engines > 0:
                    malicious_ratio = malicious / total_engines
                    suspicious_ratio = suspicious / total_engines
                    raw_score = min(malicious_ratio + suspicious_ratio * 0.5, 1.0)
                else:
                    raw_score = 0.0

                return {
                    "is_malicious": is_malicious,
                    "raw_score": round(raw_score, 2),
                    "detected_count": malicious,
                    "total_engines": total_engines,
                    "status": "completed"
                }

            except httpx.HTTPStatusError as e:
                logger.error(f"[VirusTotal] API 에러 ({e.response.status_code}): {str(e)}")
                return default_result
            except httpx.TimeoutException:
                logger.error("[VirusTotal] API 요청 타임아웃 발생")
                return default_result
            except Exception as e:
                logger.error(f"[VirusTotal] 연동 중 비정상 에러 발생: {str(e)}")
                return default_result
