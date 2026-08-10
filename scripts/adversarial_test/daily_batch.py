import argparse
import asyncio
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from app.analysis.service import SmishingAnalysisService
from scripts.adversarial_test.evaluate import ORIGINAL_LABEL, EvalResult, _evaluate_one
from scripts.adversarial_test.mutations import MUTATIONS

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_CORPUS_PATH = OUTPUT_DIR / "corpus.json"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "results.csv"

# LLM 기반 변형은 변형 생성(1콜) + 파이프라인 평가(1콜) = 2콜, 그 외는 파이프라인 평가만 1콜.
# Gemini 무료 티어 일일 한도(모델당 20건)를 며칠에 걸쳐 나눠 쓰기 위한 하루치 예산 산정에 사용.
_GEMINI_MUTATION_NAMES = {"urgency_softening", "tone_normalization", "shortening", "phone_call_redirect"}


def _estimate_cost(mutation_type: str) -> int:
    if mutation_type == ORIGINAL_LABEL:
        return 1
    return 2 if mutation_type in _GEMINI_MUTATION_NAMES else 1


def _load_done(results_path: Path) -> set[tuple[str, str]]:
    if not results_path.exists():
        return set()
    with results_path.open(encoding="utf-8", newline="") as f:
        return {(row["sample_id"], row["mutation_type"]) for row in csv.DictReader(f)}


def _append_result(result: EvalResult, out_path: Path) -> None:
    is_new = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalResult.__dataclass_fields__))
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(result))


# 이미 results.csv에 성공 기록된 (샘플, 변형) 조합은 건너뛰고, 남은 대기열에서 예산이 허용하는 만큼만
# 오늘치로 처리한다 — 프로세스가 중간에 죽어도(429 등) append 방식이라 진행분은 보존됨
async def run_daily_batch(
    corpus: list[dict],
    results_path: Path,
    budget: int,
    mutation_names: list[str] | None = None,
) -> list[EvalResult]:
    names = mutation_names if mutation_names is not None else list(MUTATIONS)
    done = _load_done(results_path)

    queue = [
        (sample, mutation_type)
        for sample in corpus
        for mutation_type in [ORIGINAL_LABEL, *names]
        if (sample["id"], mutation_type) not in done
    ]

    if not queue:
        logger.info("[DailyBatch] 남은 작업 없음 - 전체 매트릭스 완료")
        return []

    selected: list[tuple[dict, str]] = []
    spent = 0
    for sample, mutation_type in queue:
        cost = _estimate_cost(mutation_type)
        if spent + cost > budget:
            break
        selected.append((sample, mutation_type))
        spent += cost

    logger.info(
        "[DailyBatch] 오늘 처리 대상: %d건 (예상 호출 %d/%d), 이후 남는 대기열 %d건",
        len(selected),
        spent,
        budget,
        len(queue) - len(selected),
    )

    service = SmishingAnalysisService()
    semaphore = asyncio.Semaphore(1)

    results: list[EvalResult] = []
    for sample, mutation_type in selected:
        result = await _evaluate_one(service, sample, mutation_type, semaphore)
        if result is not None:
            results.append(result)
            _append_result(result, results_path)

    logger.info("[DailyBatch] 이번 배치 결과 %d/%d건 성공 저장", len(results), len(selected))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini 일일 무료 할당량 내에서 변형 평가 매트릭스를 하루치씩 이어서 채운다"
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--budget",
        type=int,
        default=18,
        help="오늘 소비할 예상 Gemini 호출 수 상한 (일일 한도 20보다 낮게 잡아 여유분 확보 권장)",
    )
    parser.add_argument("--mutations", type=str, default=None, help="쉼표 구분 변형 이름 (미지정 시 전체 10종)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    names = args.mutations.split(",") if args.mutations else None

    asyncio.run(run_daily_batch(corpus, args.results, args.budget, names))


if __name__ == "__main__":
    main()
