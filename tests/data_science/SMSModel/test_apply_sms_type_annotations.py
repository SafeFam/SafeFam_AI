"""그룹 단위 taxonomy 적용 테스트"""
from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.apply_sms_type_annotations import (
    apply_annotations,
)
from data_science.SMSModel.template_grouping import (
    create_text_fingerprint,
)


def _fingerprint(text: str) -> str:
    return create_text_fingerprint(
        normalize_text(text)
    )


def test_applies_group_type_to_all_matching_rows() -> None:
    first = "송장번호 123 주소불일치"
    second = "송장번호 456 주소불일치"

    dataset = pd.DataFrame(
        [
            {
                "text": first,
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            },
            {
                "text": second,
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            },
            {
                # 완전 중복 행도 함께 변경
                "text": second,
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            },
        ]
    )

    annotations = pd.DataFrame(
        [
            {
                "template_group_id": "group-1",
                "member_fingerprints": (
                    f"{_fingerprint(first)}|"
                    f"{_fingerprint(second)}"
                ),
                "proposed_type": "택배배송사칭",
                "review_status": "APPROVED",
                "reviewer": "tester",
                "review_note": "배송 주소 오류 유도",
            }
        ]
    )

    updated, report = apply_annotations(
        dataset,
        annotations,
    )

    assert set(updated["type"]) == {
        "택배배송사칭"
    }
    assert set(updated["label"]) == {
        "phishing"
    }
    assert report["changed_row_count"] == 3
    assert report[
        "matched_unique_fingerprint_count"
    ] == 2


def test_rejects_pending_annotations() -> None:
    dataset = pd.DataFrame(
        [
            {
                "text": "테스트",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            }
        ]
    )

    annotations = pd.DataFrame(
        [
            {
                "template_group_id": "group-1",
                "member_fingerprints": _fingerprint(
                    "테스트"
                ),
                "proposed_type": "",
                "review_status": "PENDING",
                "reviewer": "",
                "review_note": "",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="must be reviewed",
    ):
        apply_annotations(
            dataset,
            annotations,
        )