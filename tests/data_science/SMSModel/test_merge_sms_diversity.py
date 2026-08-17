"""검수된 다양성 데이터 병합 테스트."""

import pandas as pd
import pytest

from data_science.SMSModel.merge_sms_diversity import merge_approved_additions


def _row(text: str, *, status: str = "APPROVED") -> dict:
    return {
        "text": text, "label": "normal", "type": "일상대화", "has_url": False,
        "source": "user_added", "review_status": status, "reviewer": "tester",
        "review_note": "reviewed",
    }


def _dataset_row(text: str) -> dict:
    row = _row(text)
    return {key: value for key, value in row.items() if key not in {
        "review_status", "reviewer", "review_note"
    }}


def test_merge_adds_only_new_approved_fingerprints():
    dataset = pd.DataFrame([_dataset_row("기존 문장")])
    additions = pd.DataFrame([_row("기존 문장"), _row("새 문장")])

    merged, report = merge_approved_additions(dataset, additions)

    assert merged["text"].tolist() == ["기존 문장", "새 문장"]
    assert report["added_count"] == 1
    assert report["duplicate_with_dataset_count"] == 1


def test_merge_rejects_pending_review():
    dataset = pd.DataFrame([_dataset_row("기존")])
    additions = pd.DataFrame([_row("신규", status="PENDING")])

    with pytest.raises(ValueError, match="must be reviewed"):
        merge_approved_additions(dataset, additions)
