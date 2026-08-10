import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.analysis.service import SmishingAnalysisService
from scripts.adversarial_test.daily_batch import _append_result, _load_done
from scripts.adversarial_test.evaluate import EvalResult, _evaluate_one

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_CORPUS_PATH = OUTPUT_DIR / "corpus.json"
DEFAULT_MANUAL_PATH = OUTPUT_DIR / "manual_mutations.json"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "results.csv"


# 챗 앱 등으로 미리 만들어둔 변형 텍스트를 평가한다 — 변형 생성 콜은 이미 안 쓴 상태이므로
# 남은 Gemini 예산은 파이프라인 평가(항목당 1콜)에만 쓰면 된다
async def run_import(
    corpus: list[dict],
    manual_mutations: dict[str, dict[str, str | None]],
    results_path: Path,
    budget: int,
) -> list[EvalResult]:
    done = _load_done(results_path)
    by_id = {sample["id"]: sample for sample in corpus}

    queue: list[tuple[dict, str, str]] = []
    for sample_id, mutations in manual_mutations.items():
        sample = by_id.get(sample_id)
        if sample is None:
            logger.warning("[Import] 코퍼스에 없는 샘플 ID 건너뜀: %s", sample_id)
            continue
        for mutation_type, text in mutations.items():
            if not text or (sample_id, mutation_type) in done:
                continue
            queue.append((sample, mutation_type, text))

    selected = queue[:budget]
    logger.info(
        "[Import] 오늘 처리 대상: %d건 (전체 대기열 %d건 중, 예산 %d)",
        len(selected),
        len(queue),
        budget,
    )

    service = SmishingAnalysisService()
    semaphore = asyncio.Semaphore(1)

    results: list[EvalResult] = []
    for sample, mutation_type, text in selected:
        result = await _evaluate_one(service, sample, mutation_type, semaphore, text_override=text)
        if result is not None:
            results.append(result)
            _append_result(result, results_path)

    logger.info("[Import] 이번 배치 결과 %d/%d건 성공 저장", len(results), len(selected))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="수동으로 생성한 변형 텍스트를 파이프라인 평가에 임포트")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--budget",
        type=int,
        default=18,
        help="오늘 소비할 예상 Gemini 호출 수 상한 (일일 한도 20보다 낮게 잡아 여유분 확보 권장)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    manual = json.loads(args.manual.read_text(encoding="utf-8"))

    asyncio.run(run_import(corpus, manual, args.results, args.budget))


if __name__ == "__main__":
    main()
