"""검수된 SMS 다양성 데이터를 기본 데이터셋에 병합"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.data_quality.validation import validate_sms_dataset
from data_science.SMSModel.template_grouping import create_text_fingerprint

SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
SMS_DATA_DIRECTORY = SMS_MODEL_DIRECTORY.parent / "Data" / "SMSData"
DEFAULT_DATASET_PATH = SMS_DATA_DIRECTORY / "phishing_total_dataset_reclassified.csv"
DEFAULT_ADDITIONS_PATH = SMS_DATA_DIRECTORY / "sms_diversity_additions_v2.csv"
DEFAULT_REPORT_PATH = SMS_MODEL_DIRECTORY / "reports" / "sms_diversity_merge_v2.json"

BASE_SOURCES = {
    "original", "user_added", "synthetic_new_holdout", "synthetic_fp_stress",
    "synthetic_fp_stress_train", "reviewed_reclassification_v2",
    "public_phishing_v2", "synthetic_diversity_v2", "synthetic_hard_negative_v2",
    "synthetic_normal_v3", "real_holdout", "real_collected_v4",
    "real_holdout_v5", "real_phishing_v6",
}
ADDITION_SOURCES = {
    "public_phishing_v2", "synthetic_diversity_v2",
    "synthetic_hard_negative_v2", "user_added", "synthetic_normal_v3",
    "real_collected_v4", "real_holdout_v5", "real_phishing_v6",
}
REVIEW_COLUMNS = {"review_status", "reviewer", "review_note"}


def _fingerprint(text: str) -> str:
    return create_text_fingerprint(normalize_text(text))


def merge_approved_additions(
    dataset: pd.DataFrame,
    additions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """APPROVED 행만 fingerprint 중복 제거 후 병합"""
    missing = REVIEW_COLUMNS - set(additions.columns)
    
    if missing:
        raise ValueError(f"addition review columns are missing: {missing}")
    
    review_status = additions["review_status"].fillna("").astype(str).str.strip()
    if review_status.isin({"", "PENDING"}).any():
        raise ValueError("all diversity additions must be reviewed")

    approved = additions.loc[review_status.eq("APPROVED")].copy()

    if approved.empty:
        raise ValueError("no approved diversity additions")
    
    if approved["reviewer"].fillna("").str.strip().eq("").any():
        raise ValueError("approved additions require reviewer")
    
    if approved["review_note"].fillna("").str.strip().eq("").any():
        raise ValueError("approved additions require review_note")

    existing = validate_sms_dataset(dataset, allowed_sources=BASE_SOURCES)
    approved = validate_sms_dataset(approved, allowed_sources=ADDITION_SOURCES)
    existing_fingerprints = set(existing["text"].map(_fingerprint))
    approved["__fingerprint"] = approved["text"].map(_fingerprint)
    duplicate_within = int(approved["__fingerprint"].duplicated().sum())
    approved = approved.drop_duplicates(subset="__fingerprint", keep="first")
    duplicate_mask = approved["__fingerprint"].isin(existing_fingerprints)
    new_rows = approved.loc[~duplicate_mask, existing.columns].copy()
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = validate_sms_dataset(merged, allowed_sources=BASE_SOURCES)

    report = {
        "schema_version": 1,
        "reviewed_row_count": int(len(additions)),
        "approved_addition_count": int(len(approved)),
        "duplicate_within_additions_count": duplicate_within,
        "duplicate_with_dataset_count": int(duplicate_mask.sum()),
        "added_count": int(len(new_rows)),
        "final_dataset_row_count": int(len(merged)),
        "added_label_distribution": {
            str(key): int(value)
            for key, value in new_rows["label"].value_counts().sort_index().items()
        },
        "added_source_distribution": {
            str(key): int(value)
            for key, value in new_rows["source"].value_counts().sort_index().items()
        },
    }
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge approved SMS diversity rows.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--additions", type=Path, default=DEFAULT_ADDITIONS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    output_path = arguments.dataset if arguments.overwrite else arguments.output
    if output_path is None:
        raise ValueError("provide --output or use --overwrite")

    dataset = pd.read_csv(arguments.dataset)
    additions = pd.read_csv(arguments.additions, encoding="utf-8-sig")
    merged, report = merge_approved_additions(dataset, additions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[Diversity merge] added={report['added_count']} "
        f"final_rows={report['final_dataset_row_count']}"
    )
    print(f"[Diversity merge] output={output_path}")
    print(f"[Diversity merge] report={arguments.report}")


if __name__ == "__main__":
    main()
