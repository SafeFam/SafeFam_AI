"""SMS 데이터 분할 검증 및 통계 보고서 생성"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_science.SMSModel.dataset_splitting import (
    DatasetSplitConfig,
    DatasetSplits,
    validate_dataset_splits,
)
from data_science.SMSModel.template_grouping import (
    TemplateGroupingConfig,
)

REPORT_SCHEMA_VERSION = 1

REQUIRED_REPORT_COLUMNS = {
    "text_fingerprint",
    "template_group_id",
    "label",
    "type",
    "split",
}


def calculate_dataset_fingerprint(
    source: pd.DataFrame,
) -> str:
    """전체 학습 후보 데이터셋을 나타내는 SHA-256 fingerprint를 생성"""
    required_columns = {
        "text_fingerprint",
        "template_group_id",
        "label",
        "type",
    }
    missing = required_columns - set(source.columns)

    if missing:
        raise ValueError(
            f"cannot calculate dataset fingerprint; missing columns: {missing}"
        )

    canonical_rows: list[str] = []

    sorted_source = source.sort_values(
        by=[
            "text_fingerprint",
            "template_group_id",
            "label",
            "type",
        ]
    )

    for row in sorted_source[
        [
            "text_fingerprint",
            "template_group_id",
            "label",
            "type",
        ]
    ].itertuples(index=False, name=None):
        # JSON 직렬화를 사용해 문자열 결합 구분자가 실제 데이터와 충돌하는 문제를 방지
        canonical_rows.append(
            json.dumps(
                [str(value) for value in row],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    canonical_dataset = "\n".join(canonical_rows)

    return hashlib.sha256(canonical_dataset.encode("utf-8")).hexdigest()


def _label_statistics(
    df: pd.DataFrame,
) -> dict[str, dict[str, int | float]]:
    """normal/phishing별 건수와 비율을 반환"""
    result: dict[str, dict[str, int | float]] = {}

    for label in ("normal", "phishing"):
        count = int((df["label"] == label).sum())

        result[label] = {
            "count": count,
            "ratio": round(count / len(df), 6) if len(df) else 0.0,
        }

    return result


def _type_statistics(
    df: pd.DataFrame,
) -> dict[str, dict[str, int | float]]:
    """메시지 type별 건수와 split 내부 비율을 반환"""
    counts = df["type"].astype(str).value_counts()

    return {
        message_type: {
            "count": int(count),
            "ratio": round(int(count) / len(df), 6) if len(df) else 0.0,
        }
        for message_type, count in counts.sort_index().items()
    }


def _split_statistics(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """단일 split의 건수, 그룹 수, 클래스 분포와 유형 분포를 계산"""
    group_sizes = df["template_group_id"].value_counts()

    return {
        "row_count": len(df),
        "group_count": int(df["template_group_id"].nunique()),
        "largest_group_size": (int(group_sizes.max()) if not group_sizes.empty else 0),
        "labels": _label_statistics(df),
        "types": _type_statistics(df),
    }


def _find_pairwise_overlaps(
    splits: DatasetSplits,
    *,
    column: str,
) -> dict[str, dict[str, Any]]:
    """train/validation/test 사이의 그룹 또는 fingerprint 교차를 계산"""
    named_splits = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    result: dict[str, dict[str, Any]] = {}

    for left_name, right_name in pairs:
        left_values = set(named_splits[left_name][column].astype(str))
        right_values = set(named_splits[right_name][column].astype(str))

        overlap = sorted(left_values & right_values)

        result[f"{left_name}_vs_{right_name}"] = {
            "count": len(overlap),
            # 오류가 발생했을 때 보고서나 로그가 지나치게 커지지 않도록 앞의 10개만 예시로 남김
            "examples": overlap[:10],
        }

    return result


def _total_overlap_count(
    overlap_result: dict[str, dict[str, Any]],
) -> int:
    """교차 검증 결과의 전체 중복 건수를 계산"""
    return sum(int(pair_result["count"]) for pair_result in overlap_result.values())


def build_dataset_split_summary(
    source: pd.DataFrame,
    splits: DatasetSplits,
    *,
    split_config: DatasetSplitConfig,
    grouping_config: TemplateGroupingConfig,
) -> dict[str, Any]:
    """검증을 수행하고 데이터 분할 통계 보고서 dictionary를 생성"""

    validate_dataset_splits(
        source,
        splits,
        config=split_config,
    )

    group_overlaps = _find_pairwise_overlaps(
        splits,
        column=split_config.group_column,
    )
    fingerprint_overlaps = _find_pairwise_overlaps(
        splits,
        column=split_config.fingerprint_column,
    )

    total_row_count = len(source)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset_fingerprint": calculate_dataset_fingerprint(source),
        "configuration": {
            "template_grouping": {
                "similarity_threshold": (grouping_config.similarity_threshold),
                "ngram_range": list(grouping_config.ngram_range),
                "min_df": grouping_config.min_df,
                "max_features": grouping_config.max_features,
            },
            "dataset_split": {
                "train_size": split_config.train_size,
                "validation_size": split_config.val_size,
                "test_size": split_config.test_size,
                "random_state": split_config.random_state,
                "candidate_count": (split_config.candidate_count),
            },
        },
        "validation": {
            "passed": True,
            "group_overlap_count": _total_overlap_count(group_overlaps),
            "fingerprint_overlap_count": (_total_overlap_count(fingerprint_overlaps)),
            "group_overlaps": group_overlaps,
            "fingerprint_overlaps": fingerprint_overlaps,
        },
        "dataset": {
            "row_count": int(total_row_count),
            "group_count": int(source["template_group_id"].nunique()),
            "labels": _label_statistics(source),
            "types": _type_statistics(source),
        },
        "splits": {
            "train": {
                **_split_statistics(splits.train),
                "dataset_ratio": round(
                    len(splits.train) / total_row_count,
                    6,
                ),
            },
            "validation": {
                **_split_statistics(splits.validation),
                "dataset_ratio": round(
                    len(splits.validation) / total_row_count,
                    6,
                ),
            },
            "test": {
                **_split_statistics(splits.test),
                "dataset_ratio": round(
                    len(splits.test) / total_row_count,
                    6,
                ),
            },
        },
    }


def render_dataset_split_markdown(
    summary: dict[str, Any],
) -> str:
    """JSON summary를 사람이 검토하기 쉬운 Markdown 문서로 변환"""
    lines = [
        "# SMS Dataset Split Summary",
        "",
        "## Dataset",
        "",
        f"- Schema version: `{summary['schema_version']}`",
        (f"- Dataset fingerprint: `{summary['dataset_fingerprint']}`"),
        (f"- Total rows: {summary['dataset']['row_count']}"),
        (f"- Template groups: {summary['dataset']['group_count']}"),
        "",
        "## Configuration",
        "",
        (
            "- Template similarity threshold: "
            f"`{summary['configuration']['template_grouping']['similarity_threshold']}`"
        ),
        (
            "- Template n-gram range: "
            f"`{summary['configuration']['template_grouping']['ngram_range']}`"
        ),
        (
            "- Random state: "
            f"`{summary['configuration']['dataset_split']['random_state']}`"
        ),
        "",
        "## Split Overview",
        "",
        "| Split | Rows | Ratio | Groups | Normal | Phishing |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split_name in ("train", "validation", "test"):
        split = summary["splits"][split_name]

        lines.append(
            f"| {split_name} "
            f"| {split['row_count']} "
            f"| {split['dataset_ratio']:.2%} "
            f"| {split['group_count']} "
            f"| {split['labels']['normal']['count']} "
            f"({split['labels']['normal']['ratio']:.2%}) "
            f"| {split['labels']['phishing']['count']} "
            f"({split['labels']['phishing']['ratio']:.2%}) |"
        )

    validation = summary["validation"]

    lines.extend(
        [
            "",
            "## Leakage Validation",
            "",
            f"- Passed: `{validation['passed']}`",
            (f"- Template group overlap count: `{validation['group_overlap_count']}`"),
            (
                "- Fingerprint overlap count: "
                f"`{validation['fingerprint_overlap_count']}`"
            ),
            "",
            "## Message Type Distribution",
            "",
        ]
    )

    for split_name in ("train", "validation", "test"):
        lines.extend(
            [
                f"### {split_name.title()}",
                "",
                "| Type | Count | Ratio |",
                "|---|---:|---:|",
            ]
        )

        for message_type, statistics in summary["splits"][split_name]["types"].items():
            # type 값에 |가 포함되면 Markdown table이 깨지므로 escape
            escaped_type = message_type.replace("|", "\\|")

            lines.append(
                f"| {escaped_type} "
                f"| {statistics['count']} "
                f"| {statistics['ratio']:.2%} |"
            )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_dataset_split_reports(
    summary: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """JSON과 Markdown 보고서를 저장"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown_path.write_text(
        render_dataset_split_markdown(summary),
        encoding="utf-8",
    )


def generate_dataset_split_reports(
    source: pd.DataFrame,
    splits: DatasetSplits,
    *,
    split_config: DatasetSplitConfig,
    grouping_config: TemplateGroupingConfig,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """검증 → summary 생성 → JSON/Markdown 저장을 한 번에 실행"""
    summary = build_dataset_split_summary(
        source,
        splits,
        split_config=split_config,
        grouping_config=grouping_config,
    )

    save_dataset_split_reports(
        summary,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    return summary
