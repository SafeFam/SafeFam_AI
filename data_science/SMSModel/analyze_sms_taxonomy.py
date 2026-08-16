"""기타피싱 검수용 annotation CSV와 분포 보고서 생성"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.analysis.text.preprocessing import normalize_text
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

DEFAULT_REPORT_PATH = (
    SMS_MODEL_DIRECTORY
    / "reports"
    / "sms_taxonomy_audit_v2.json"
)

def build_annotation_rows(
        dataset: pd.DataFrame,
) -> pd.DataFrame:
    """기타피싱만 익명 fingerprint 기준 검수 목록으로 변환"""

    targets = dataset[
        (dataset["label"] == "phishing")
        & (dataset["type"] == "기타피싱")
    ].copy()

    targets["text_norm"] = targets["text"].map(
        normalize_text
    )
    targets["text_fingerprint"] = targets[
        "text_norm"
    ].map(create_text_fingerprint)

    return targets[
        [
            "text_fingerprint",
            "text",
            "label",
            "type",
            "source",
        ]
    ].assign(
        proposed_type="",
        review_status="PENDING",
        reviewer="",
        review_note="",
    )

def build_audit_report(
        dataset: pd.DataFrame,
        annotations: pd.DataFrame,
) -> dict:
    """재분류 전 분포와 검수 상태를 기록"""

    return {
        "schema_version": 1,
        "dataset_row_count": len(dataset),
        "other_phishing_count": len(annotations),
        "type_distribution": {
            str(key): int(value)
            for key, value in dataset[
                ["label", "type"]
            ].value_counts().sort_index().items()
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in annotations[
                "source"
            ].value_counts().sort_index().items()
        },
        "review_status": {
            str(key): int(value)
            for key, value in annotations[
                "review_status"
            ].value_counts().items()
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    arguments = parser.parse_args()

    dataset = pd.read_csv(arguments.dataset)
    annotations = build_annotation_rows(dataset)

    arguments.annotations.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    annotations.to_csv(
        arguments.annotations,
        index=False,
        encoding="utf-8-sig",
    )

    report = build_audit_report(
        dataset,
        annotations,
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


if __name__ == "__main__":
    main()

