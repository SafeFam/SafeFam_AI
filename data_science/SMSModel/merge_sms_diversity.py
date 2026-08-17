import pandas as pd
from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.data_quality.validation import validate_sms_dataset
from data_science.SMSModel.template_grouping.fingerprint import create_text_fingerprint

def merge_approved_additions(
    dataset: pd.DataFrame,
    additions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    approved = additions[
        additions["review_status"] == "APPROVED"
    ].copy()

    if approved.empty:
        raise ValueError(
            "no approved diversity additions"
        )

    approved = validate_sms_dataset(
        approved,
        allowed_sources={
            "public_phishing_v2",
            "synthetic_diversity_v2",
            "synthetic_hard_negative_v2",
            "user_added",
        },
    )

    existing = dataset.copy()
    existing["__fingerprint"] = existing[
        "text"
    ].map(
        lambda text: create_text_fingerprint(
            normalize_text(text)
        )
    )

    approved["__fingerprint"] = approved[
        "text"
    ].map(
        lambda text: create_text_fingerprint(
            normalize_text(text)
        )
    )

    approved = approved.drop_duplicates(
        subset="__fingerprint"
    )

    duplicate_mask = approved[
        "__fingerprint"
    ].isin(existing["__fingerprint"])

    new_rows = approved[
        ~duplicate_mask
    ].copy()

    merged = pd.concat(
        [
            existing.drop(
                columns="__fingerprint"
            ),
            new_rows[
                existing.drop(
                    columns="__fingerprint"
                ).columns
            ],
        ],
        ignore_index=True,
    )

    report = {
        "approved_addition_count": len(approved),
        "duplicate_skipped_count": int(
            duplicate_mask.sum()
        ),
        "added_count": len(new_rows),
    }

    return merged, report