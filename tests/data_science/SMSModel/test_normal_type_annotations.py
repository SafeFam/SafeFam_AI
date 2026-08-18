"""정상 문자 세부 유형 재분류(#83) 경로 테스트"""
from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.analyze_sms_taxonomy import (
    build_annotation_rows,
    suggest_normal_type,
)
from data_science.SMSModel.apply_sms_type_annotations import (
    apply_annotations,
)
from data_science.SMSModel.template_grouping import (
    create_text_fingerprint,
)


def _fingerprint(text: str) -> str:
    return create_text_fingerprint(normalize_text(text))


def make_mixed_dataset() -> pd.DataFrame:
    """정상 두 유형과 피싱 한 건이 섞인 데이터셋"""
    return pd.DataFrame(
        [
            {
                "text": "[국민은행] 출금 50,000원 잔액 120,000원",
                "label": "normal",
                "type": "기타정상",
                "has_url": False,
                "source": "original",
            },
            {
                "text": "오늘 저녁에 뭐 먹을까 고민중",
                "label": "normal",
                "type": "일상대화",
                "has_url": False,
                "source": "original",
            },
            {
                "text": "송장번호 123 주소불일치 확인 바랍니다",
                "label": "phishing",
                "type": "기타피싱",
                "has_url": False,
                "source": "original",
            },
        ]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[NICE] 인증번호 [123456]를 입력해주세요", "정상인증알림"),
        ("고객님의 상품이 배송완료 되었습니다", "정상택배배송안내"),
        ("[국민은행] 출금 50,000원 잔액 120,000원", "정상금융알림"),
        ("주민센터 민원 처리 안내드립니다", "정상공공기관알림"),
        ("오늘 저녁에 뭐 먹을까", "기타정상"),
    ],
)
def test_suggests_normal_type_from_keywords(text: str, expected: str) -> None:
    """정상 문자 유형 제안 규칙이 대표 표현을 잡아내는지 검증"""
    suggested_type, reason = suggest_normal_type(text)

    assert suggested_type == expected
    assert reason


def test_builds_annotations_for_normal_rows_only() -> None:
    """target_label=normal이면 피싱 행이 검수 대상에서 빠져야 한다"""
    annotations = build_annotation_rows(
        make_mixed_dataset(),
        target_label="normal",
        current_types=("기타정상", "일상대화"),
    )

    # 정상 2건만 그룹으로 잡히고 피싱 1건은 제외된다.
    assert int(annotations["unique_member_count"].sum()) == 2
    assert "기타피싱" not in set(annotations["current_type"])


def test_rejects_unknown_target_type() -> None:
    """대상 type이 데이터에 없으면 명확히 실패해야 한다"""
    with pytest.raises(ValueError, match="does not contain"):
        build_annotation_rows(
            make_mixed_dataset(),
            target_label="normal",
            current_types=("존재하지않는유형",),
        )


def test_applies_normal_annotations_without_touching_phishing() -> None:
    """정상 행만 새 유형으로 바뀌고 피싱 행과 label은 그대로여야 한다"""
    dataset = make_mixed_dataset()
    banking_text = "[국민은행] 출금 50,000원 잔액 120,000원"

    annotations = pd.DataFrame(
        [
            {
                "template_group_id": "group-1",
                "member_fingerprints": _fingerprint(banking_text),
                "proposed_type": "정상금융알림",
                "review_status": "APPROVED",
                "reviewer": "tester",
                # APPROVED 행은 reviewer와 review_note가 모두 채워져야 한다.
                "review_note": "은행 발신 입출금 알림",
            }
        ]
    )

    updated, report = apply_annotations(
        dataset,
        annotations,
        target_label="normal",
        current_types=("기타정상", "일상대화"),
    )

    assert set(updated.loc[updated["label"] == "normal", "type"]) == {
        "정상금융알림",
        "일상대화",
    }
    # 피싱 행과 이진 label은 재분류의 영향을 받지 않아야 한다.
    assert updated.loc[updated["label"] == "phishing", "type"].tolist() == [
        "기타피싱"
    ]
    assert updated["label"].tolist() == dataset["label"].tolist()
    assert report["changed_row_count"] == 1


def test_rejects_annotations_crossing_label_boundary() -> None:
    """정상 검수인데 피싱 행을 가리키면 거부해야 한다"""
    dataset = make_mixed_dataset()
    phishing_text = "송장번호 123 주소불일치 확인 바랍니다"

    annotations = pd.DataFrame(
        [
            {
                "template_group_id": "group-1",
                "member_fingerprints": _fingerprint(phishing_text),
                "proposed_type": "정상금융알림",
                "review_status": "APPROVED",
                "reviewer": "tester",
                # APPROVED 행은 reviewer와 review_note가 모두 채워져야 한다.
                "review_note": "은행 발신 입출금 알림",
            }
        ]
    )

    # 피싱 행은 current_types에 없으므로 대상 자체가 잡히지 않는다.
    with pytest.raises(ValueError, match="do not match dataset rows"):
        apply_annotations(
            dataset,
            annotations,
            target_label="normal",
            current_types=("기타정상", "일상대화"),
        )
