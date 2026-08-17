"""그룹 단위 SMS taxonomy audit 테스트"""
from __future__ import annotations

import pandas as pd

from data_science.SMSModel.analyze_sms_taxonomy import (
    build_annotation_rows,
    build_audit_report,
    suggest_message_type,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": "송장번호 123 주소불일치 링크 확인",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": True,
                "source": "original",
            },
            {
                # 숫자만 다른 유사 템플릿
                "text": "송장번호 456 주소불일치 링크 확인",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": True,
                "source": "original",
            },
            {
                # 완전 중복
                "text": "송장번호 456 주소불일치 링크 확인",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": True,
                "source": "original",
            },
            {
                "text": "급등주 종목추천 리딩방 참여",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            },
            {
                # annotation 대상이 아님
                "text": "정상 카드 결제 안내",
                "label": "normal",
                "type": "정상카드결제알림",
                "has_url": False,
                "source": "original",
            },
        ]
    )


def test_suggests_delivery_phishing() -> None:
    message_type, reason = suggest_message_type(
        "송장번호 확인 주소불일치 링크"
    )

    assert message_type == "택배배송사칭"
    assert reason


def test_builds_one_annotation_per_template_group() -> None:
    annotations = build_annotation_rows(
        _dataset()
    )

    assert len(annotations) < 4
    assert (
        annotations["template_group_id"].nunique()
        == len(annotations)
    )
    assert (
        annotations["raw_member_count"].sum()
        == 4
    )
    assert annotations[
        "member_fingerprints"
    ].notna().all()


def test_audit_report_tracks_raw_and_group_counts() -> None:
    dataset = _dataset()
    annotations = build_annotation_rows(
        dataset
    )

    report = build_audit_report(
        dataset,
        annotations,
    )

    assert report["raw_other_phishing_count"] == 4
    assert report["annotation_group_count"] == len(
        annotations
    )
    assert report["covered_raw_row_count"] == 4