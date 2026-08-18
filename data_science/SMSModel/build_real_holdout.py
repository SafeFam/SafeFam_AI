"""실제 문자로 구성한 주 평가셋(real_holdout)을 분리한다.

#83까지 holdout 210건은 전부 합성 데이터라 실제 성능을 판정할 수 없었다. 이
스크립트는 학습 pool의 실제 문자 중 일부를 template group 단위로 떼어 평가
전용으로 태깅한다.

원칙
- template group 전체를 통째로 옮겨 학습셋과의 누수를 막는다.
- 합성 행이 한 건이라도 섞인 group은 선택하지 않는다(평가셋 오염 방지).
- label과 세부 유형 비율을 유지하되, 표본이 적은 유형은 학습에 남긴다.
- 같은 fingerprint를 가진 원본 중복 행은 함께 이동한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.template_grouping import (
    create_text_fingerprint,
    prepare_template_groups,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    HOLDOUT_SOURCES,
    build_template_grouping_config,
)

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = (
    SMS_MODEL_DIRECTORY / "reports" / "real_holdout_build.json"
)

REAL_HOLDOUT_SOURCE = "real_holdout"

# 평가셋에 넣지 않을 source 접두사. 합성 데이터로 평가하면 성능이 부풀려진다.
SYNTHETIC_SOURCE_PREFIX = "synthetic"

# 이 비율만큼 실제 문자를 평가셋으로 뗀다.
DEFAULT_HOLDOUT_RATIO = 0.20

# 유형별 표본이 이 값보다 적으면 전부 학습에 남긴다.
MINIMUM_TYPE_SIZE = 10

# 재현을 위해 그룹 선택 순서를 고정한다.
RANDOM_STATE = 42


def load_candidate_pool(dataset: pd.DataFrame) -> pd.DataFrame:
    """학습 pool을 만들고 template group을 부여한다."""
    working = dataset.copy()
    working["text_norm"] = working["text"].map(normalize_text)

    source = working.get("source", pd.Series("original", index=working.index))
    pool = working[~source.isin(HOLDOUT_SOURCES)].reset_index(drop=True)

    return prepare_template_groups(
        pool,
        config=build_template_grouping_config(),
        text_column="text_norm",
        label_column="label",
    )


def select_holdout_groups(
    pool: pd.DataFrame,
    *,
    holdout_ratio: float,
    minimum_type_size: int,
) -> set[str]:
    """평가셋으로 옮길 template group id를 고른다."""
    is_synthetic = (
        pool["source"].astype(str).str.startswith(SYNTHETIC_SOURCE_PREFIX)
    )

    # 합성 행이 하나라도 포함된 group은 통째로 후보에서 뺀다.
    contaminated = set(pool.loc[is_synthetic, "template_group_id"])

    group_summary = (
        pool[~pool["template_group_id"].isin(contaminated)]
        .groupby("template_group_id")
        .agg(
            row_count=("text_fingerprint", "size"),
            label=("label", "first"),
            type=("type", "first"),
        )
        .reset_index()
    )

    selected: set[str] = set()

    # (label, type)별로 같은 비율만큼 떼어 평가셋의 구성이 학습셋과 닮게 만든다.
    for (label, type_name), stratum in group_summary.groupby(["label", "type"]):
        available_rows = int(stratum["row_count"].sum())
        if available_rows < minimum_type_size:
            # 표본이 적은 유형까지 떼면 학습에서 그 유형이 사라진다.
            continue

        target_rows = available_rows * holdout_ratio

        # 큰 group부터 담으면 목표를 넘기기 쉬우므로 작은 group부터 채운다.
        ordered = stratum.sort_values(
            ["row_count", "template_group_id"],
            ascending=[True, True],
        )

        taken = 0
        for row in ordered.itertuples(index=False):
            if taken + row.row_count > target_rows:
                continue
            selected.add(str(row.template_group_id))
            taken += int(row.row_count)

    return selected


def apply_holdout_tag(
    dataset: pd.DataFrame,
    pool: pd.DataFrame,
    holdout_groups: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """선택된 group의 모든 원본 행에 real_holdout source를 부여한다."""
    holdout_fingerprints = set(
        pool.loc[
            pool["template_group_id"].isin(holdout_groups),
            "text_fingerprint",
        ].astype(str)
    )

    result = dataset.copy()
    result["__fingerprint"] = result["text"].map(
        lambda text: create_text_fingerprint(normalize_text(text))
    )

    target_mask = result["__fingerprint"].isin(holdout_fingerprints)
    if not target_mask.any():
        raise ValueError("selected groups do not match any dataset row")

    previous_sources = (
        result.loc[target_mask, "source"].astype(str).value_counts().to_dict()
    )
    original_labels = result["label"].copy()

    result.loc[target_mask, "source"] = REAL_HOLDOUT_SOURCE

    if not result["label"].equals(original_labels):
        raise RuntimeError("labels changed while tagging holdout")

    moved = result.loc[target_mask]
    report = {
        "moved_row_count": int(target_mask.sum()),
        "moved_unique_count": len(holdout_fingerprints),
        "moved_group_count": len(holdout_groups),
        "previous_sources": {str(k): int(v) for k, v in previous_sources.items()},
        "label_distribution": {
            str(k): int(v) for k, v in moved["label"].value_counts().items()
        },
        "type_distribution": {
            str(k): int(v)
            for k, v in moved["type"].value_counts().sort_index().items()
        },
    }

    return result.drop(columns="__fingerprint"), report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag real messages as the primary evaluation holdout."
    )
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=DEFAULT_HOLDOUT_RATIO,
    )
    parser.add_argument(
        "--minimum-type-size",
        type=int,
        default=MINIMUM_TYPE_SIZE,
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    output_path = (
        arguments.dataset if arguments.overwrite else arguments.output
    )
    if output_path is None:
        raise ValueError("provide --output or use --overwrite")

    dataset = pd.read_csv(arguments.dataset)

    if (dataset.get("source") == REAL_HOLDOUT_SOURCE).any():
        raise ValueError(
            "dataset already contains real_holdout rows; "
            "revert before rebuilding"
        )

    pool = load_candidate_pool(dataset)
    holdout_groups = select_holdout_groups(
        pool,
        holdout_ratio=arguments.holdout_ratio,
        minimum_type_size=arguments.minimum_type_size,
    )
    updated, report = apply_holdout_tag(dataset, pool, holdout_groups)

    report["holdout_ratio"] = arguments.holdout_ratio
    report["minimum_type_size"] = arguments.minimum_type_size
    report["candidate_pool_unique_count"] = int(len(pool))

    updated.to_csv(output_path, index=False, encoding="utf-8-sig")

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "[Real holdout] "
        f"groups={report['moved_group_count']} "
        f"unique={report['moved_unique_count']} "
        f"rows={report['moved_row_count']}"
    )
    print(f"[Real holdout] labels={report['label_distribution']}")
    print(f"[Real holdout] output={output_path}")
    print(f"[Real holdout] report={arguments.report}")


if __name__ == "__main__":
    main()
