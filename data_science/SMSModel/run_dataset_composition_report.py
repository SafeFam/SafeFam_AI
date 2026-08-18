"""데이터셋 구성 진단 (유형별 보유량과 split 분포 보고서)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_science.SMSModel.dataset_splitting import DatasetSplits
from data_science.SMSModel.train_sms import DATA_PATH, load_data, split_data

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SMS_MODEL_DIRECTORY / "reports" / "dataset_composition.json"

MINIMUM_TOTAL_FOR_COVERAGE = 5

def count_by_type(
    frame: pd.DataFrame, 
    label: str
) -> dict[str, int]:
    """지정한 label의 유형별 건수를 많은 순서대로 반환"""
    counts = frame.loc[frame["label"] == label, "type"].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def count_by_split(
    splits: DatasetSplits, 
    label: str
) -> dict[str, dict[str, int]]:
    """split별 유형 분포 반환"""
    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {name: count_by_type(frame, label) for name, frame in frames.items()}

def find_uncovered_types(
    types_by_split: dict[str, dict[str, int]],
    *,
    minimum_total: int = MINIMUM_TOTAL_FOR_COVERAGE,
) -> list[dict[str, object]]:
    """보유량이 충분한데도 특정 split에 한 건도 없는 유형 찾기"""
    all_types: set[str] = set()
    for counts in types_by_split.values():
        all_types.update(counts)

    findings: list[dict[str, object]] = []
    for type_name in sorted(all_types):
        per_split = {
            split: counts.get(type_name, 0)
            for split, counts in types_by_split.items()
        }
        total = sum(per_split.values())
        empty_splits = [split for split, count in per_split.items() if count == 0]

        if total >= minimum_total and empty_splits:
            findings.append(
                {
                    "type": type_name,
                    "total": total,
                    "per_split": per_split,
                    "missing_in": empty_splits,
                }
            )
    return findings

def count_values(
    frame: pd.DataFrame,
    column: str
) -> dict[str, int]:
    """컬럼이 없으면 빈 dict를 반환해 CSV 스키마 변화에 대응"""
    if column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts().items()}

def build_report() -> dict[str, object]:
    """원본 pool,holdout,split을 한 번에 훑어 구성 보고서 생성"""
    raw = pd.read_csv(DATA_PATH)
    pool, holdout = load_data(DATA_PATH)

    splits = split_data(pool, create_manifest=False)

    phishing_by_split = count_by_split(splits, "phishing")
    normal_by_split = count_by_split(splits, "normal")

    return {
        "raw_row_count": int(len(raw)),
        "raw_source_counts": count_values(raw, "source"),
        "pool_row_count": int(len(pool)),
        "pool_label_counts": count_values(pool, "label"),
        "holdout_row_count": int(len(holdout)),
        "holdout_source_counts": count_values(holdout, "source"),
        "normal_type_counts": count_by_type(pool, "normal"),
        "phishing_type_counts": count_by_type(pool, "phishing"),
        "phishing_types_by_split": phishing_by_split,
        "normal_types_by_split": normal_by_split,
        "phishing_types_missing_in_a_split": find_uncovered_types(phishing_by_split),
        "normal_types_missing_in_a_split": find_uncovered_types(normal_by_split),
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose SMS dataset composition and split distributions."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    report = build_report()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[Report] {arguments.output}")
    print(f"  pool={report['pool_row_count']} rows {report['pool_label_counts']}")
    for finding in report["phishing_types_missing_in_a_split"]:
        print(
            f"  [Imbalance] {finding['type']} (total: {finding['total']}) -> "
            f"missing in: {', '.join(finding['missing_in'])}"
        )


if __name__ == "__main__":
    main()