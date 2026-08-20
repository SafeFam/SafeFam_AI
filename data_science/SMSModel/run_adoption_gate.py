"""단독 운영 채택 기준으로 artifact를 판정"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_science.SMSModel.evaluation.adoption import (
    AdoptionCriteria,
    evaluate_adoption,
)
from data_science.SMSModel.evaluation.stacking_reporting import (
    predict_probabilities_with_latency,
)
from data_science.SMSModel.evaluation.standalone_bands import (
    BandEdgesUnreachableError,
    select_standalone_bands,
    summarize_bands,
)
from data_science.SMSModel.run_stacking_training import (
    STACKING_MODEL_PATH,
)
from data_science.SMSModel.run_error_analysis import load_classifier
from data_science.SMSModel.run_standalone_analysis import (
    SELECTION_SPLIT,
    collect_scored_splits,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    load_data,
    select_real_holdout,
    select_synthetic_stress,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = STACKING_MODEL_PATH
DEFAULT_OUTPUT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "adoption_gate.json"
)

JUDGING_SPLIT = "real_holdout"


def fingerprint_judging_set(frame: pd.DataFrame) -> str:
    """판정 split의 구성이 바뀌지 않았는지 확인할 지문"""
    canonical = "\n".join(sorted(frame["text"].astype(str)))

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_report(classifier) -> dict[str, object]:
    """경계 선정부터 판정까지 한 번에 수행"""
    criteria = AdoptionCriteria()
    scored = collect_scored_splits(classifier)

    try:
        edges = select_standalone_bands(
            *scored[SELECTION_SPLIT],
            max_alert_false_positive_rate=(
                criteria.max_alert_false_positive_rate
            ),
            min_coverage_recall=criteria.min_coverage_recall,
        )
    except BandEdgesUnreachableError as error:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reason": str(error),
            "selection_split": SELECTION_SPLIT,
            "judging_split": JUDGING_SPLIT,
        }

    judged = summarize_bands(*scored[JUDGING_SPLIT], edges)

    _, holdout = load_data(DATA_PATH)
    _, _, latencies_ms = predict_probabilities_with_latency(
        classifier,
        select_real_holdout(holdout),
    )
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    assessment = evaluate_adoption(
        judged,
        p95_latency_ms=p95_latency_ms,
        criteria=criteria,
    )

    synthetic = select_synthetic_stress(holdout)

    judging_frame = select_real_holdout(holdout)

    return {
        "selection_split": SELECTION_SPLIT,
        "judging_split": JUDGING_SPLIT,
        "judging_set": {
            "sample_count": int(len(judging_frame)),
            "fingerprint": fingerprint_judging_set(judging_frame),
        },
        "edges": edges.to_dict(),
        "criteria": AdoptionCriteria().__dict__,
        "measured": judged,
        "p95_latency_ms": p95_latency_ms,
        "assessment": assessment.to_dict(),
        "synthetic_reference": {
            "sample_count": int(len(synthetic)),
            "note": "합성 스트레스 셋은 참고용이며 판정에 사용하지 않는다",
        },
    }


def print_summary(report: dict[str, object]) -> None:
    """판정 결과를 사람이 읽을 형태로 출력"""
    assessment = report.get("assessment")

    if assessment is None:
        print(f"[Adoption] {report['verdict']} — {report['reason']}")
        return

    print(f"[Adoption] 판정 split={report['judging_split']}")
    edges = report["edges"]
    print(
        f"  경계 normal_max={edges['normal_max']:.4f}"
        f" phishing_min={edges['phishing_min']:.4f}"
        f" ({report['selection_split']}에서 선정)"
    )

    marks = {
        "PASS": "통과",
        "FAIL": "미달",
        "INSUFFICIENT_EVIDENCE": "판정불가",
    }
    for criterion in assessment["criteria"]:
        print(
            f"    {marks[criterion['outcome']]:<5}"
            f" {criterion['name']:<28}"
            f" 측정 {criterion['measured']:.4g}"
            f" / 기준 {criterion['required']:.4g}"
        )

    print(f"\n  종합 판정: {assessment['verdict']}")


def main() -> None:
    """CLI 인자를 읽어 채택 기준 판정 리포트를 JSON으로 저장"""
    parser = argparse.ArgumentParser(
        description="Judge a stacking artifact against the adoption criteria."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    classifier = load_classifier(arguments.model_path)
    report = build_report(classifier)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[Adoption] {arguments.output}")
    print_summary(report)


if __name__ == "__main__":
    main()
