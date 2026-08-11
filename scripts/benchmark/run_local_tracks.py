import argparse
import asyncio
import csv
import json
import logging
import time
from pathlib import Path

from app.analysis.rules.analyzer import analyze_text_with_rules
from app.analysis.text.naive_bayes_analyzer import analyze_text_with_naive_bayes
from scripts.benchmark.corpus import DEFAULT_OUTPUT_PATH as DEFAULT_CORPUS_PATH
from scripts.benchmark.metrics import compute_metrics, format_table

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DETECTION_THRESHOLD = 40  # scoring.py의 MEDIUM 임계값과 동일 기준 (SUSPICIOUS 이상을 탐지로 간주)


def _run_rules(corpus: list[dict]) -> list[dict]:
    rows = []
    for sample in corpus:
        start = time.perf_counter()
        result = analyze_text_with_rules(sample["text"])
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "track": "규칙만",
                "score": result["rule_score"],
                "detected": result["rule_score"] >= DETECTION_THRESHOLD,
                "elapsed_seconds": elapsed,
            }
        )
    return rows


async def _run_naive_bayes(corpus: list[dict]) -> list[dict]:
    # 모델 최초 로드 시간(1회성)이 평균에 섞이지 않도록 워밍업 호출을 먼저 수행
    await analyze_text_with_naive_bayes("워밍업")

    rows = []
    for sample in corpus:
        start = time.perf_counter()
        result = await analyze_text_with_naive_bayes(sample["text"])
        elapsed = time.perf_counter() - start
        risk_score = result["result"]["risk_score"]
        rows.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "track": "NB만",
                "score": risk_score,
                "detected": risk_score >= DETECTION_THRESHOLD,
                "elapsed_seconds": elapsed,
            }
        )
    return rows


def _write_raw(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "track", "score", "detected", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="규칙엔진/나이브베이즈 단독 트랙 벤치마크 (API 호출 없음)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "local_tracks_raw.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    logger.info("[LocalTracks] 코퍼스 %d건 (phishing %d / normal %d)", len(corpus),
                sum(1 for c in corpus if c["label"] == "phishing"),
                sum(1 for c in corpus if c["label"] == "normal"))

    rule_rows = _run_rules(corpus)
    nb_rows = asyncio.run(_run_naive_bayes(corpus))

    all_rows = rule_rows + nb_rows
    _write_raw(all_rows, args.out)

    rule_metrics = compute_metrics("규칙만", rule_rows)
    nb_metrics = compute_metrics("NB만", nb_rows)

    table = format_table([rule_metrics, nb_metrics])
    print(table)

    summary_path = OUTPUT_DIR / "local_tracks_summary.md"
    summary_path.write_text(table + "\n", encoding="utf-8")
    logger.info("[LocalTracks] 요약 -> %s", summary_path)


if __name__ == "__main__":
    main()
