import argparse
import asyncio
import csv
import inspect
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from app.analysis.rules.analyzer import analyze_text_with_rules
from app.analysis.service import SmishingAnalysisService
from scripts.adversarial_test.mutations import MUTATIONS
from scripts.adversarial_test.rate_limit import LLM_RATE_LIMITER

logger = logging.getLogger(__name__)

# scoring.py의 등급 임계값(>=70 HIGH, >=40 MEDIUM)과 동일 기준을 규칙 단독 트랙에도 적용해
# "규칙엔진만 썼다면 탐지했을까"를 파이프라인 판정(LOW 아님)과 같은 잣대로 비교한다.
RULE_DETECTION_THRESHOLD = 40

OUTPUT_DIR = Path(__file__).parent / "output"
ORIGINAL_LABEL = "ORIGINAL"


@dataclass
class EvalResult:
    sample_id: str
    mutation_type: str
    text: str
    rule_score: int
    rule_detected: bool
    pipeline_grade: str
    pipeline_score: int
    pipeline_detected: bool


async def _apply_mutation(name: str, text: str) -> str:
    mutate = MUTATIONS[name]
    if inspect.iscoroutinefunction(mutate):
        return await mutate(text)
    return mutate(text)


async def _evaluate_one(
    service: SmishingAnalysisService,
    sample: dict,
    mutation_type: str,
    semaphore: asyncio.Semaphore,
    text_override: str | None = None,
) -> EvalResult | None:
    async with semaphore:
        try:
            # text_override: 챗 앱 등으로 이미 만들어둔 변형 텍스트를 그대로 평가할 때 생성 단계를 건너뜀
            if text_override is not None:
                text = text_override
            elif mutation_type == ORIGINAL_LABEL:
                text = sample["text"]
            else:
                text = await _apply_mutation(mutation_type, sample["text"])
            rule_result = analyze_text_with_rules(text)
            # 파이프라인 내부에서도 대부분 LLM을 호출하므로(자체 모델 확신 구간은 예외)
            # 변형 생성용 호출과 같은 전역 리미터로 최소 간격을 강제한다
            await LLM_RATE_LIMITER.wait()
            pipeline = await service.analyze_pipeline(text)
        except Exception as exception:
            logger.error(
                "[Evaluate] 평가 실패 - sample=%s mutation=%s error_type=%s",
                sample["id"],
                mutation_type,
                type(exception).__name__,
            )
            return None

    # 파이프라인 내부 폴백은 실패를 MEDIUM으로 반환하므로, 그대로 집계하면 탐지율이 부풀려진다
    if pipeline.status != "SUCCESS":
        logger.error(
            "[Evaluate] 파이프라인 오류로 제외 - sample=%s mutation=%s",
            sample["id"],
            mutation_type,
        )
        return None

    rule_score = rule_result["rule_score"]
    grade = getattr(pipeline.risk_grade, "value", pipeline.risk_grade)
    logger.info(
        "[Evaluate] %s/%s rule=%d pipeline=%d(%s)",
        sample["id"],
        mutation_type,
        rule_score,
        pipeline.final_score,
        grade,
    )
    return EvalResult(
        sample_id=sample["id"],
        mutation_type=mutation_type,
        text=text,
        rule_score=rule_score,
        rule_detected=rule_score >= RULE_DETECTION_THRESHOLD,
        pipeline_grade=grade,
        pipeline_score=pipeline.final_score,
        pipeline_detected=grade != "LOW",
    )


async def evaluate_corpus(
    corpus: list[dict],
    mutation_names: list[str] | None = None,
    concurrency: int = 5,
) -> list[EvalResult]:
    names = mutation_names if mutation_names is not None else list(MUTATIONS)
    unknown = [n for n in names if n not in MUTATIONS]
    if unknown:
        raise ValueError(f"알 수 없는 변형 유형: {unknown}")

    service = SmishingAnalysisService()
    # Gemini rate limit 보호: 변형 생성 + 파이프라인 분석 모두 LLM을 태우므로 동시 실행 수를 제한한다
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _evaluate_one(service, sample, mutation_type, semaphore)
        for sample in corpus
        for mutation_type in [ORIGINAL_LABEL, *names]
    ]
    logger.info("[Evaluate] 총 %d건 평가 시작 (동시성 %d)", len(tasks), concurrency)

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def write_results_csv(results: list[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    logger.info("[Evaluate] 결과 %d건 저장 -> %s", len(results), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="사기문자 변형 공격에 대한 탐지 성능 평가")
    parser.add_argument("--corpus", type=Path, default=OUTPUT_DIR / "corpus.json")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "results.csv")
    parser.add_argument("--mutations", type=str, default=None, help="쉼표 구분 변형 이름 (미지정 시 전체)")
    parser.add_argument("--limit", type=int, default=None, help="코퍼스 앞 N건만 평가 (비용 통제용)")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if args.limit:
        corpus = corpus[: args.limit]

    names = args.mutations.split(",") if args.mutations else None
    results = asyncio.run(evaluate_corpus(corpus, names, args.concurrency))
    write_results_csv(results, args.out)


if __name__ == "__main__":
    main()
