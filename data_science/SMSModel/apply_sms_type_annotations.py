"""승인된 그룹 annotation을 SMS 원본 데이터에 적용"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.data_quality import (
    NORMAL_MESSAGE_TYPES,
    PHISHING_MESSAGE_TYPES,
    normalize_message_types,
)
from data_science.SMSModel.template_grouping import (
    create_text_fingerprint,
)


SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_DATASET_PATH = (
    SMS_MODEL_DIRECTORY.parent
    / "Data"
    / "SMSData"
    / "phishing_total_dataset_reclassified.csv"
)

DEFAULT_ANNOTATION_PATH = (
    SMS_MODEL_DIRECTORY.parent
    / "Data"
    / "SMSData"
    / "sms_type_annotations_v2.csv"
)

DEFAULT_CHANGE_REPORT_PATH = (
    SMS_MODEL_DIRECTORY
    / "reports"
    / "sms_taxonomy_changes_v2.json"
)


REQUIRED_ANNOTATION_COLUMNS = {
    "template_group_id",
    "member_fingerprints",
    "proposed_type",
    "review_status",
    "reviewer",
    "review_note",
}


def _split_fingerprints(
    value: str,
) -> set[str]:
    """CSV 한 셀에 저장된 fingerprint 목록 복원"""

    if not isinstance(value, str):
        raise ValueError(
            "member_fingerprints must be a string"
        )

    fingerprints = {
        item.strip()
        for item in value.split("|")
        if item.strip()
    }

    if not fingerprints:
        raise ValueError(
            "annotation group has no member fingerprints"
        )

    return fingerprints


def allowed_types_for_label(target_label: str) -> frozenset[str]:
    """검수 대상 label에서 proposed_type으로 허용되는 유형 집합"""

    if target_label == "normal":
        return NORMAL_MESSAGE_TYPES

    return PHISHING_MESSAGE_TYPES


def validate_approved_annotations(
    annotations: pd.DataFrame,
    *,
    # 기본값은 #77의 기타피싱 검수와 동일하게 유지한다.
    allowed_types: frozenset[str] = PHISHING_MESSAGE_TYPES,
) -> pd.DataFrame:
    """APPROVED annotation의 필수값과 중복을 검증"""

    missing = (
        REQUIRED_ANNOTATION_COLUMNS
        - set(annotations.columns)
    )

    if missing:
        raise ValueError(
            f"annotation columns are missing: {missing}"
        )

    if (
        annotations["review_status"] == "PENDING"
    ).any():
        raise ValueError(
            "all annotation groups must be reviewed"
        )

    approved = annotations[
        annotations["review_status"] == "APPROVED"
    ].copy()

    if approved.empty:
        raise ValueError(
            "no approved annotations are available"
        )

    if approved["template_group_id"].duplicated().any():
        raise ValueError(
            "approved annotations contain duplicate groups"
        )

    if approved["proposed_type"].isna().any():
        raise ValueError(
            "approved annotations require proposed_type"
        )

    unsupported = (
        set(approved["proposed_type"].astype(str))
        - allowed_types
    )

    if unsupported:
        raise ValueError(
            f"unsupported proposed types: "
            f"{sorted(unsupported)}"
        )

    if approved["reviewer"].fillna("").str.strip().eq("").any():
        raise ValueError(
            "approved annotations require reviewer"
        )

    if approved[
        "review_note"
    ].fillna("").str.strip().eq("").any():
        raise ValueError(
            "approved annotations require review_note"
        )

    # 서로 다른 그룹이 같은 fingerprint를 소유하면 한 메시지에 두 type이 적용될 수 있으므로 차단
    all_fingerprints: list[str] = []

    for value in approved["member_fingerprints"]:
        all_fingerprints.extend(
            _split_fingerprints(str(value))
        )

    if len(all_fingerprints) != len(
        set(all_fingerprints)
    ):
        raise ValueError(
            "fingerprint belongs to multiple approved groups"
        )

    return approved


def apply_annotations(
    dataset: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    # 기본값은 #77의 기타피싱 검수와 동일하게 유지한다.
    target_label: str = "phishing",
    current_types: tuple[str, ...] = ("기타피싱",),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """승인된 그룹을 원본 행에 적용하고 변경 보고서를 반환"""

    approved = validate_approved_annotations(
        annotations,
        allowed_types=allowed_types_for_label(target_label),
    )
    result = normalize_message_types(dataset)

    original_labels = result["label"].copy()

    result["__text_norm"] = result["text"].map(normalize_text)
    result["__fingerprint"] = result["__text_norm"].map(create_text_fingerprint)

    fingerprint_to_type: dict[str, str] = {}
    fingerprint_to_group: dict[str, str] = {}

    for row in approved.itertuples(index=False):
        fingerprints = _split_fingerprints(str(row.member_fingerprints))
        for fingerprint in fingerprints:
            fingerprint_to_type[fingerprint] = str(row.proposed_type)
            fingerprint_to_group[fingerprint] = str(row.template_group_id)

    target_mask = (
        result["__fingerprint"].isin(fingerprint_to_type)
        & (result["type"].isin(current_types))
    )

    if not target_mask.any():
        raise ValueError("approved annotations do not match dataset rows")

    # 검수 대상 label 밖의 행이 수정되면 이진 정답이 오염되므로 차단한다.
    if (result.loc[target_mask, "label"] != target_label).any():
        raise ValueError(
            f"annotations must only modify rows labelled {target_label}"
        )

    before_types = (
        result.loc[target_mask, "type"].astype(str).value_counts().to_dict()
    )

    result.loc[target_mask, "type"] = result.loc[
        target_mask, "__fingerprint"
    ].map(fingerprint_to_type)

    changed_rows = int(target_mask.sum())

    if not result["label"].equals(original_labels):
        raise RuntimeError("binary labels changed during annotation")

    matched_fingerprints = set(
        result.loc[target_mask, "__fingerprint"].astype(str)
    )
    
    already_resolved_fingerprints = set(
        result.loc[
            result["__fingerprint"].isin(fingerprint_to_type)
            & (~result["type"].isin(current_types)),
            "__fingerprint",
        ].astype(str)
    )
    
    expected_fingerprints = set(fingerprint_to_type)
    missing_fingerprints = (
        expected_fingerprints - matched_fingerprints - already_resolved_fingerprints
    )

    if missing_fingerprints:
        raise ValueError(
            "approved annotation fingerprints are missing "
            f"from dataset: {sorted(missing_fingerprints)[:10]}"
        )

    change_report = {
        "schema_version": 1,
        "approved_group_count": len(approved),
        "changed_row_count": changed_rows,
        "matched_unique_fingerprint_count": len(matched_fingerprints),
        "before_type_distribution": {
            str(key): int(value) for key, value in before_types.items()
        },
        "after_type_distribution": {
            str(key): int(value)
            for key, value in result.loc[target_mask, "type"]
            .value_counts()
            .sort_index()
            .items()
        },
        "unchanged_binary_label_count": len(result),
    }

    return (
        result.drop(columns=["__text_norm", "__fingerprint"]),
        change_report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply approved SMS type annotations."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATION_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_CHANGE_REPORT_PATH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--label",
        default="phishing",
        choices=("phishing", "normal"),
        help="검수 대상 label",
    )
    parser.add_argument(
        "--current-type",
        nargs="+",
        default=["기타피싱"],
        help="검수 대상 type (여러 개 지정 가능)",
    )
    arguments = parser.parse_args()

    output_path = (
        arguments.dataset
        if arguments.overwrite
        else arguments.output
    )

    if output_path is None:
        raise ValueError(
            "provide --output or use --overwrite"
        )

    dataset = pd.read_csv(arguments.dataset)
    annotations = pd.read_csv(
        arguments.annotations,
        encoding="utf-8-sig",
    )

    updated, report = apply_annotations(
        dataset,
        annotations,
        target_label=arguments.label,
        current_types=tuple(arguments.current_type),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    updated.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    arguments.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    arguments.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[Taxonomy apply] "
        f"groups={report['approved_group_count']} "
        f"changed_rows={report['changed_row_count']}"
    )
    print(
        f"[Taxonomy apply] output={output_path}"
    )
    print(
        f"[Taxonomy apply] report={arguments.report}"
    )


if __name__ == "__main__":
    main()