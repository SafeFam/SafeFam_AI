"""고정된 test split의 Claude 예측 수집 및 오프라인 평가"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from app.analysis.hybrid_policy import (
    ConditionalLlmPolicy,
    HybridThresholds,
)
from app.analysis.text.stacking_analyzer import (
    analyze_text_with_stacking,
)
from app.core.config import settings
from data_science.SMSModel.hybrid_evaluation import (
    EvaluationMode,
    EvaluationSample,
    HybridEvaluationRunner,
)
from data_science.SMSModel.hybrid_evaluation.cache import (
    EVALUATION_SCHEMA_VERSION,
    ClaudeTestCache,
    build_prompt_version,
    calculate_dataset_fingerprint,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    SPLIT_MANIFEST_PATH,
    load_data,
    split_data,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent

STACKING_METADATA_PATH = (
    SMS_MODEL_DIRECTORY
    / "artifacts"
    / "stacking"
    / "metadata.json"
)

HYBRID_POLICY_REPORT_PATH = (
    SMS_MODEL_DIRECTORY
    / "artifacts"
    / "stacking"
    / "hybrid_policy.json"
)

CLAUDE_TEST_CACHE_PATH = (
    SMS_MODEL_DIRECTORY
    / "artifacts"
    / "stacking"
    / "llm_test_predictions.json"
)

EVALUATION_RECORDS_PATH = (
    SMS_MODEL_DIRECTORY
    / "reports"
    / "hybrid_evaluation"
    / "evaluation_records.json"
)


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

def load_test_data():
    """저장된 manifest를 사용해 test spli만 반환"""

    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: {SPLIT_MANIFEST_PATH}"
        )

    dataset, _unused_holdout = load_data(
        DATA_PATH
    )

    splits = split_data(
        dataset,
        create_manifest=False,
    )

    test = splits.test.copy()

    required_columns = {
        "text",
        "label",
        "text_fingerprint",
    }

    missing = required_columns - set(test.columns)

    if missing:
        raise ValueError(
            f"test columns are missing: {missing}"
        )

    if test.empty:
        raise ValueError(
            "test split must not be empty"
        )

    if test["text_fingerprint"].duplicated().any():
        raise ValueError(
            "test fingerprints must be unique"
        )

    if set(test["label"].astype(str)) - {
        "normal",
        "phishing",
    }:
        raise ValueError(
            "test split contains unsupported labels"
        )

    return test

def calculate_test_dataset_fingerprint(test) -> str:
    return calculate_dataset_fingerprint(
        (
            str(row.text_fingerprint),
            str(row.label),
        )
        for row in test.itertuples(index=False)
    )

def load_frozen_validation_policy() -> ConditionalLlmPolicy:
    """validation에서 선정된 임계값만 읽음"""

    if not STACKING_METADATA_PATH.is_file():
        raise FileNotFoundError(
            "stacking metadata is required"
        )

    if not HYBRID_POLICY_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "validation hybrid policy report is required"
        )

    metadata = json.loads(
        STACKING_METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    report = json.loads(
        HYBRID_POLICY_REPORT_PATH.read_text(encoding="utf-8")
    )

    if report.get("source_split") != "validation":
        raise ValueError(
            "hybrid policy must be selected from validation split"
        )

    if report.get("split_manifest") != SPLIT_MANIFEST_PATH.name:
        raise ValueError("hybrid policy split manifest mismatch")

    policy = metadata.get("hybrid_policy")

    if not isinstance(policy, dict):
        raise RuntimeError(
            "validation-selected hybrid_policy is missing from "
            "stacking metadata; run validation threshold selection first"
        )

    if report.get("selection") != policy:
        raise ValueError(
            "hybrid policy report and stacking metadata mismatch"
        )

    normal_max = policy.get(
        "normal_probability_max"
    )
    phishing_min = policy.get(
        "phishing_probability_min"
    )

    if (
        isinstance(normal_max, bool)
        or not isinstance(normal_max, (int, float))
        or not math.isfinite(float(normal_max))
    ):
        raise ValueError(
            "validation normal threshold is invalid"
        )

    if (
        isinstance(phishing_min, bool)
        or not isinstance(phishing_min, (int, float))
        or not math.isfinite(float(phishing_min))
    ):
        raise ValueError(
            "validation phishing threshold is invalid"
        )

    return ConditionalLlmPolicy(
        HybridThresholds(
            normal_max=float(normal_max),
            phishing_min=float(phishing_min),
        )
    )

async def collect_missing_predictions(
        test,
        *,
        cache: ClaudeTestCache,
        delay_seconds: float,
) -> None:
    """누락됐거나 실패한 test 예측만 다시 수집"""

    if settings.MOCK_SECURITY_API:
        raise RuntimeError(
            "MOCK_SECURITY_API must be false during test collection"
        )

    if delay_seconds < 0:
        raise ValueError(
            "delay_seconds must not be negative"
        )

    # offline 모드에서는 이 import 경로에 도달 X
    from app.analysis.text.llm_analyzer import (
        analyze_text_with_llm,
    )

    total = len(test)

    for position, row in enumerate(
        test.itertuples(index=False),
        start=1,
    ):
        fingerprint = str(
            row.text_fingerprint
        )

        # 성공한 호출만 영구적으로 건너뜀
        # 실패한 항목은 다음 실행에서 다시 시도
        if cache.contains(
            fingerprint,
            require_success=True,
        ):
            print(
                f"[Claude test] cached {position}/{total}"
            )
            continue

        analysis = await analyze_text_with_llm(
            str(row.text)
        )

        cache.store_analysis(
            text_fingerprint=fingerprint,
            analysis=analysis,
        )

        cache.save()

        available = cache.entries[
            fingerprint
        ]["available"]

        print(
            f"[Claude test] collected {position}/{total} "
            f"available={available}"
        )

        if delay_seconds > 0:
            await asyncio.sleep(
                delay_seconds
            )

def build_samples(test) -> list[EvaluationSample]:
    """원문은 메모리의 EvaluationSample에만 존재"""
    return [
        EvaluationSample(
            sample_id=str(row.text_fingerprint),
            text=str(row.text),
            expected_label=str(row.label),
        )
        for row in test.itertuples(index=False)
    ]

def build_cached_llm_analyzer(
    *,
    cache: ClaudeTestCache,
    fingerprint_by_text: dict[str, str],
):
    """HybridEvaluationRunner의 text 기반 인터페이스를 캐시에 연결"""

    async def analyze(text: str) -> dict[str, Any]:
        fingerprint = fingerprint_by_text.get(text)

        if fingerprint is None:
            raise KeyError(
                "message does not belong to the fixed test split"
            )

        return await cache.analyze_cached(
            fingerprint
        )

    return analyze


async def run_offline_evaluation(
    test,
    *,
    cache: ClaudeTestCache,
    policy: ConditionalLlmPolicy,
) -> list:
    """Bedrock을 호출하지 않고 캐시만 사용해 세 모드를 실행"""
    expected_fingerprints = {
        str(value)
        for value in test["text_fingerprint"]
    }

    cache.require_complete(
        expected_fingerprints
    )

    fingerprint_by_text = {
        str(row.text): str(row.text_fingerprint)
        for row in test.itertuples(index=False)
    }

    if len(fingerprint_by_text) != len(test):
        raise ValueError(
            "test split contains duplicate message text"
        )

    cached_llm_analyzer = build_cached_llm_analyzer(
        cache=cache,
        fingerprint_by_text=fingerprint_by_text,
    )

    runner = HybridEvaluationRunner(
        policy=policy,
        stacking_analyzer=analyze_text_with_stacking,
        llm_analyzer=cached_llm_analyzer,
    )

    return await runner.evaluate(
        build_samples(test),
        modes=(
            EvaluationMode.SELF_MODEL_ONLY,
            EvaluationMode.LLM_ONLY,
            EvaluationMode.HYBRID,
        ),
    )


def save_evaluation_records(
    records,
    *,
    dataset_fingerprint: str,
) -> None:
    """원문 없는 평가 레코드를 저장"""
    _atomic_write_json(
        EVALUATION_RECORDS_PATH,
        {
            "evaluation_schema_version": (
                EVALUATION_SCHEMA_VERSION
            ),
            "source_split": "test",
            "dataset_fingerprint": dataset_fingerprint,
            "model_id": settings.BEDROCK_MODEL_ID,
            "region": settings.AWS_REGION,
            "prompt_version": build_prompt_version(),
            "record_count": len(records),
            "records": [
                record.to_dict()
                for record in records
            ],
        },
    )


async def async_main(
    *,
    collect: bool,
    offline: bool,
    delay_seconds: float,
) -> None:
    if collect and offline:
        raise ValueError(
            "--collect and --offline cannot be used together"
        )

    test = load_test_data()

    # 비용이 발생하는 test 호출 전에 validation 정책을 먼저 검증합니다.
    policy = load_frozen_validation_policy()

    dataset_fingerprint = (
        calculate_test_dataset_fingerprint(test)
    )

    cache = ClaudeTestCache(
        path=CLAUDE_TEST_CACHE_PATH,
        dataset_fingerprint=dataset_fingerprint,
        model_id=settings.BEDROCK_MODEL_ID,
        region=settings.AWS_REGION,
        prompt_version=build_prompt_version(),
    )

    cache.load()

    if collect:
        await collect_missing_predictions(
            test,
            cache=cache,
            delay_seconds=delay_seconds,
        )

    # --offline 또는 수집 완료 후 모두 캐시만 사용
    records = await run_offline_evaluation(
        test,
        cache=cache,
        policy=policy,
    )

    save_evaluation_records(
        records,
        dataset_fingerprint=dataset_fingerprint,
    )

    print(
        "[Hybrid evaluation] completed "
        f"samples={len(test)} "
        f"records={len(records)}"
    )

    print(
        f"[Hybrid evaluation] cache={CLAUDE_TEST_CACHE_PATH}"
    )

    print(
        f"[Hybrid evaluation] records={EVALUATION_RECORDS_PATH}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Claude test predictions and run "
            "Stacking/Claude/Hybrid evaluation."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Call Bedrock only for missing or failed test predictions."
        ),
    )

    mode.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Never call Bedrock; fail if the test cache is incomplete."
        ),
    )

    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
    )

    arguments = parser.parse_args()

    # 아무 옵션도 없으면 안전한 offline 동작으로 처리
    offline = arguments.offline or not arguments.collect

    asyncio.run(
        async_main(
            collect=arguments.collect,
            offline=offline,
            delay_seconds=arguments.delay_seconds,
        )
    )


if __name__ == "__main__":
    main()

