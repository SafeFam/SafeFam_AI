import argparse
import asyncio
import csv
import json
import logging
import time
from pathlib import Path

from app.analysis.text.gemini_analyzer import analyze_text_with_gemini
from app.core.config import settings
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH
from scripts.benchmark.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUT_PATH = OUTPUT_DIR / "llm_track_raw.csv"
DETECTION_THRESHOLD = 40


# gemini-flash-latest(프로덕션 기본값)는 일일 20건 한도로 400건 전체를 못 돌리지만,
# GEMINI_MODEL 환경변수로 다른 모델(예: gemini-3.1-flash-lite, 일 500건/분당 15건)을
# 지정하면 전체 코퍼스를 실제 API로 돌릴 수 있다. 어떤 모델을 썼는지는 결과에 기록해
# 보고서에서 "프로덕션 기본 모델과 다름"을 명시할 수 있게 한다.
async def run(corpus: list[dict], rpm: int) -> list[dict]:
    limiter = RateLimiter(min_interval_seconds=60.0 / rpm)
    rows = []

    for sample in corpus:
        await limiter.wait()
        start = time.perf_counter()
        try:
            result = await analyze_text_with_gemini(sample["text"])
        except Exception as exception:  # noqa: BLE001
            logger.error("[LlmTrack] 호출 실패 id=%s error=%s", sample["id"], type(exception).__name__)
            continue
        elapsed = time.perf_counter() - start

        if result.get("result", {}).get("error_message"):
            logger.warning(
                "[LlmTrack] API 오류 응답 제외 id=%s error=%s",
                sample["id"],
                result["result"]["error_message"],
            )
            continue

        risk_score = result["result"]["risk_score"]
        rows.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "model": settings.GEMINI_MODEL,
                "score": risk_score,
                "detected": risk_score >= DETECTION_THRESHOLD,
                "elapsed_seconds": elapsed,
            }
        )
        logger.info("[LlmTrack] id=%s score=%d elapsed=%.2fs", sample["id"], risk_score, elapsed)

    return rows


def _write(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "model", "score", "detected", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM만 트랙 전체 코퍼스 실측 (GEMINI_MODEL 환경변수로 모델 지정)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--rpm", type=int, default=13, help="분당 요청 상한 (모델 RPM보다 여유있게 설정)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("[LlmTrack] 사용 모델: %s", settings.GEMINI_MODEL)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    rows = asyncio.run(run(corpus, args.rpm))
    _write(rows, args.out)
    logger.info("[LlmTrack] %d/%d건 저장 -> %s", len(rows), len(corpus), args.out)


if __name__ == "__main__":
    main()
