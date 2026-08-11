import argparse
import asyncio
import csv
import json
import logging
import random
import time
from pathlib import Path

from app.analysis.url.analyzer import HybridUrlAnalyzer
from app.analysis.url.tracker import extract_urls, trace_url
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_OUT_PATH = OUTPUT_DIR / "url_track_raw.csv"


# GSB/VT는 Gemini와 별개 쿼터라 즉시 실행 가능. VT 무료 티어 속도 제한(4req/min) 때문에
# has_url=True 전체(282건) 대신 표본을 잡아 실측하고, 결과는 SafeFam 전체 트랙 조합에 사용한다.
async def run_url_track(corpus: list[dict], sample_size: int, seed: int = 42) -> list[dict]:
    with_url = [s for s in corpus if s["has_url"]]
    rng = random.Random(seed)
    sample = rng.sample(with_url, min(sample_size, len(with_url)))

    analyzer = HybridUrlAnalyzer()
    rows = []

    for item in sample:
        urls = extract_urls(item["text"])
        if not urls:
            continue

        start = time.perf_counter()
        try:
            traced = await trace_url(urls[0])
            result = await analyzer.scan_url(traced)
        except Exception as exception:  # noqa: BLE001
            logger.error("[UrlTrack] 실패 id=%s error=%s", item["id"], type(exception).__name__)
            continue
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "is_malicious": result.get("is_malicious", False),
                "url_risk_score": result.get("url_risk_score", 0.0),
                "available": result.get("available", False),
                "elapsed_seconds": elapsed,
            }
        )
        logger.info(
            "[UrlTrack] id=%s malicious=%s risk=%.2f elapsed=%.2fs",
            item["id"],
            result.get("is_malicious"),
            result.get("url_risk_score", 0.0),
            elapsed,
        )

    return rows


def _write(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "label", "is_malicious", "url_risk_score", "available", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="URL 트랙(GSB/VT) 실측 - Gemini 쿼터와 무관")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    rows = asyncio.run(run_url_track(corpus, args.sample_size))
    _write(rows, args.out)
    logger.info("[UrlTrack] %d건 저장 -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
