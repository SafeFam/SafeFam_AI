import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from app.analysis.text.gemini_analyzer import analyze_text_with_gemini
from app.core.config import settings
from scripts.adversarial_test.rate_limit import GEMINI_RATE_LIMITER
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUT_PATH = OUTPUT_DIR / "llm_timing.json"


# LLM만 트랙의 평균 분석시간은 전체 400건을 실제 API로 돌릴 예산이 없어 소량 실측으로 대체한다.
# 정확도(Recall/FPR)는 챗 앱 프록시로 별도 확보하고, 이 스크립트는 순수 지연시간만 측정한다.
async def measure(corpus: list[dict], sample_size: int) -> dict:
    samples = corpus[:sample_size]
    elapsed_list: list[float] = []
    failures = 0

    for sample in samples:
        await GEMINI_RATE_LIMITER.wait()
        start = time.perf_counter()
        try:
            result = await analyze_text_with_gemini(sample["text"])
        except Exception as exception:  # noqa: BLE001
            logger.error("[LlmTiming] 호출 실패 id=%s error=%s", sample["id"], type(exception).__name__)
            failures += 1
            continue
        elapsed = time.perf_counter() - start

        if result.get("result", {}).get("error_message"):
            logger.warning("[LlmTiming] API 오류 응답이라 제외 id=%s", sample["id"])
            failures += 1
            continue

        elapsed_list.append(elapsed)
        logger.info("[LlmTiming] id=%s elapsed=%.2fs", sample["id"], elapsed)

    avg = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
    return {
        "model": settings.GEMINI_MODEL,
        "sample_size": sample_size,
        "success_count": len(elapsed_list),
        "failure_count": failures,
        "avg_time_seconds": avg,
        "raw_elapsed_seconds": elapsed_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM(Gemini) 트랙 평균 분석시간 소량 실측")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--sample-size", type=int, default=18)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    result = asyncio.run(measure(corpus, args.sample_size))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "[LlmTiming] 성공 %d / 실패 %d, 평균 %.2f초 -> %s",
        result["success_count"],
        result["failure_count"],
        result["avg_time_seconds"],
        args.out,
    )


if __name__ == "__main__":
    main()
