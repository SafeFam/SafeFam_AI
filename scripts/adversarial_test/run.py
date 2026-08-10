import argparse
import asyncio
import logging
from pathlib import Path

from scripts.adversarial_test.aggregate import build_summary, read_results
from scripts.adversarial_test.corpus import (
    DEFAULT_CSV_PATH,
    parse_category_overrides,
    select_corpus,
    write_corpus,
)
from scripts.adversarial_test.evaluate import evaluate_corpus, write_results_csv

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="코퍼스 생성 -> 변형 평가 -> 집계 전체 실행")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument(
        "--category-overrides",
        type=str,
        default=None,
        help="예: 금융기관사칭=25,정부공공기관사칭=10",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        help="쉼표 구분 키워드. 지정 시 텍스트에 하나라도 포함된 행만 사용 (예: 계좌,이체,정지,동결)",
    )
    parser.add_argument(
        "--min-rule-score",
        type=int,
        default=None,
        help="지정 시 규칙엔진 점수가 이 값 이상인 원문만 사용 (원본은 규칙엔진이 이미 탐지하는 케이스로 한정)",
    )
    parser.add_argument("--mutations", type=str, default=None, help="쉼표 구분 변형 이름 (미지정 시 전체)")
    parser.add_argument("--limit", type=int, default=None, help="코퍼스 앞 N건만 평가 (비용 통제용)")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = select_corpus(
        args.csv,
        per_type=args.per_type,
        category_overrides=parse_category_overrides(args.category_overrides),
        keyword_filter=args.keywords.split(",") if args.keywords else None,
        min_rule_score=args.min_rule_score,
    )
    write_corpus(corpus, OUTPUT_DIR / "corpus.json")

    if args.limit:
        corpus = corpus[: args.limit]

    names = args.mutations.split(",") if args.mutations else None
    results = asyncio.run(evaluate_corpus(corpus, names, args.concurrency))

    results_path = OUTPUT_DIR / "results.csv"
    write_results_csv(results, results_path)

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text(build_summary(read_results(results_path)), encoding="utf-8")
    logger.info("[Run] 요약 리포트 -> %s", summary_path)


if __name__ == "__main__":
    main()
