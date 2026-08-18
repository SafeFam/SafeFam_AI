"""검수 대상 type을 템플릿 그룹 단위로 묶어 annotation 생성

기본값은 #77에서 쓰던 기타피싱 검수와 동일하며, --label/--current-type으로
정상 문자 세부 유형 재분류(#83)에도 사용할 수 있다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.analysis.text.preprocessing import normalize_text
from data_science.SMSModel.template_grouping import (
    prepare_template_groups,
)
from data_science.SMSModel.train_sms import (
    build_template_grouping_config,
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


ANNOTATION_COLUMNS = [
    "template_group_id",
    "representative_text",
    "unique_member_count",
    "raw_member_count",
    "member_fingerprints",
    "current_type",
    "source_summary",
    "suggested_type",
    "suggestion_reason",
    "proposed_type",
    "review_status",
    "reviewer",
    "review_note",
]


def suggest_message_type(
    text: str,
) -> tuple[str, str]:
    """검수 편의를 위한 보수적인 type 후보를 제안"""

    normalized = normalize_text(text)

    if any(
        keyword in normalized
        for keyword in (
            "송장",
            "배송",
            "주소불일치",
            "물품보관",
            "택배",
        )
    ):
        return (
            "택배배송사칭",
            "배송·송장·주소 오류 관련 표현",
        )

    if any(
        keyword in normalized
        for keyword in (
            "리딩방",
            "급등주",
            "종목추천",
            "투자수익",
            "코인투자",
        )
    ):
        return (
            "투자리딩방사기",
            "투자·종목 추천·리딩방 관련 표현",
        )

    if any(
        keyword in normalized
        for keyword in (
            "대출승인",
            "저금리",
            "대환대출",
            "대출상담",
        )
    ):
        return (
            "대출사기",
            "대출 승인·대환·저금리 관련 표현",
        )

    if any(
        keyword in normalized
        for keyword in (
            "원격도우미",
            "앱설치",
            "어플설치",
            "apk",
        )
    ):
        return (
            "악성링크앱설치유도",
            "원격 제어 또는 앱 설치 요구",
        )

    if any(
        keyword in normalized
        for keyword in (
            "본인인증",
            "계정정지",
            "이용정지",
            "인증만료",
        )
    ):
        return (
            "계정정지본인인증유도",
            "계정 정지 또는 본인인증 요구",
        )

    if any(
        keyword in normalized
        for keyword in (
            "결제내역",
            "결제취소",
            "환불",
            "승인내역",
        )
    ):
        return (
            "결제환불사칭",
            "허위 결제·취소·환불 관련 표현",
        )

    if any(
        keyword in normalized
        for keyword in (
            "당첨",
            "경품",
            "무료쿠폰",
            "지원금대상",
        )
    ):
        return (
            "이벤트당첨사칭",
            "당첨·경품·쿠폰·지원금 수령 유도",
        )

    if any(
        keyword in normalized
        for keyword in (
            "알바",
            "부업",
            "재택근무",
            "고수익",
        )
    ):
        return (
            "채용부업사기",
            "채용·알바·부업·고수익 관련 표현",
        )

    # 자동으로 억지 분류하지 않고 검토 대상으로 남김
    return (
        "기타피싱",
        "명확한 단일 유형을 자동 제안할 수 없음",
    )


# 정상 문자 세부 유형 제안 규칙. 위에서부터 먼저 일치하는 항목이 채택되므로
# 더 구체적인 유형을 앞에 둔다. 어디까지나 검수 편의용 초안이며 최종 판단은
# 사람이 proposed_type에 직접 적는다.
NORMAL_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "정상인증알림",
        ("인증번호", "인증코드", "본인확인", "로그인"),
        "인증번호·로그인 알림 관련 표현",
    ),
    (
        "정상택배배송안내",
        ("배송완료", "배송출발", "운송장", "택배", "수령"),
        "배송 상태·운송장 관련 표현",
    ),
    (
        "정상공공기관알림",
        ("주민센터", "구청", "시청", "국세청", "예비군", "민방위", "보건소"),
        "공공기관 발신 안내 관련 표현",
    ),
    (
        "정상카드결제알림",
        ("승인", "일시불", "할부", "결제금액"),
        "카드 승인·결제 관련 표현",
    ),
    (
        "정상금융알림",
        ("출금", "입금", "잔액", "이체", "계좌"),
        "입출금·잔액·이체 관련 표현",
    ),
    (
        "정상포인트소멸알림",
        ("포인트", "마일리지", "소멸예정", "적립금"),
        "포인트·적립금 소멸 관련 표현",
    ),
    (
        "정상광고프로모션",
        ("쿠폰", "할인", "이벤트", "프로모션", "특가"),
        "쿠폰·할인·프로모션 관련 표현",
    ),
)


def suggest_normal_type(
    text: str,
) -> tuple[str, str]:
    """정상 문자의 세부 유형 후보를 보수적으로 제안"""

    normalized = normalize_text(text)

    for type_name, keywords, reason in NORMAL_TYPE_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return (type_name, reason)

    # 억지로 분류하지 않고 사람이 판단하도록 남김
    return (
        "기타정상",
        "명확한 단일 유형을 자동 제안할 수 없음",
    )


def suggest_type_for_label(
    text: str,
    *,
    target_label: str,
) -> tuple[str, str]:
    """검수 대상 label에 맞는 유형 후보를 제안"""

    if target_label == "normal":
        return suggest_normal_type(text)

    return suggest_message_type(text)


def _join_unique(
    values: pd.Series,
) -> str:
    """그룹의 고유 값을 정렬된 문자열로 결합"""

    return "|".join(
        sorted(
            {
                str(value)
                for value in values
                if str(value).strip()
            }
        )
    )


def build_annotation_rows(
    dataset: pd.DataFrame,
    *,
    # 기본값은 #77의 기타피싱 검수와 동일하게 유지한다.
    target_label: str = "phishing",
    current_types: tuple[str, ...] = ("기타피싱",),
) -> pd.DataFrame:
    """검수 대상 type을 그룹 대표 1행으로 변환"""

    required_columns = {
        "text",
        "label",
        "type",
        "source",
    }
    missing = required_columns - set(dataset.columns)

    if missing:
        raise ValueError(
            f"dataset columns are missing: {missing}"
        )

    targets = dataset[
        (dataset["label"] == target_label)
        & (dataset["type"].isin(current_types))
    ].copy()

    if targets.empty:
        raise ValueError(
            f"dataset does not contain {target_label} rows "
            f"with types {sorted(current_types)}"
        )

    targets["text_norm"] = targets["text"].map(
        normalize_text
    )

    # 완전 중복을 제거하기 전에 fingerprint별 원본 행 수를 보존
    # 같은 문자가 원본에 여러 번 있어도 그룹 검수는 한 번만 수행
    from data_science.SMSModel.template_grouping import (
        add_text_fingerprints,
    )

    fingerprinted = add_text_fingerprints(
        targets,
        text_column="text_norm",
    )

    raw_counts = (
        fingerprinted["text_fingerprint"]
        .value_counts()
        .to_dict()
    )

    grouped = prepare_template_groups(
        targets,
        config=build_template_grouping_config(),
        text_column="text_norm",
        label_column="label",
    )

    annotation_rows: list[dict[str, Any]] = []

    for group_id, group in grouped.groupby(
        "template_group_id",
        sort=True,
    ):
        # 대표 문자는 가장 긴 메시지로 선택
        representative_index = (
            group["text"]
            .astype(str)
            .str.len()
            .idxmax()
        )
        representative_text = str(
            group.loc[representative_index, "text"]
        )

        suggested_type, suggestion_reason = (
            suggest_type_for_label(
                representative_text,
                target_label=target_label,
            )
        )

        fingerprints = sorted(
            group["text_fingerprint"].astype(str)
        )

        annotation_rows.append(
            {
                "template_group_id": str(group_id),
                "representative_text": representative_text,
                "unique_member_count": len(fingerprints),
                "raw_member_count": sum(
                    raw_counts[fingerprint]
                    for fingerprint in fingerprints
                ),
                # CSV 한 셀 안에 |로 구분해 저장
                "member_fingerprints": "|".join(
                    fingerprints
                ),
                # 여러 type을 한 번에 검수할 수 있어 그룹의 실제 값을 기록한다.
                "current_type": _join_unique(group["type"]),
                "source_summary": _join_unique(
                    group["source"]
                ),
                "suggested_type": suggested_type,
                "suggestion_reason": suggestion_reason,
                # proposed_type은 사람이 확정
                "proposed_type": "",
                "review_status": "PENDING",
                "reviewer": "",
                "review_note": "",
            }
        )

    result = pd.DataFrame(
        annotation_rows,
        columns=ANNOTATION_COLUMNS,
    )

    # 그룹 수와 annotation 행 수가 반드시 같아야 함
    if len(result) != grouped[
        "template_group_id"
    ].nunique():
        raise RuntimeError(
            "annotation row count does not match template group count"
        )

    return result.sort_values(
        by=[
            "suggested_type",
            "raw_member_count",
            "template_group_id",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_audit_report(
    dataset: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    target_label: str = "phishing",
    current_types: tuple[str, ...] = ("기타피싱",),
) -> dict[str, Any]:
    """그룹 기반 검수 현황과 분포를 기록"""

    raw_target_count = int(
        (
            (dataset["label"] == target_label)
            & (dataset["type"].isin(current_types))
        ).sum()
    )

    return {
        # v3: 검수 대상이 피싱으로 고정돼 있지 않으므로 target 중립 필드를 쓴다.
        "schema_version": 3,
        "target_label": target_label,
        "current_types": sorted(current_types),
        "dataset_row_count": len(dataset),
        "raw_target_count": raw_target_count,
        "annotation_group_count": len(annotations),
        "unique_target_count": int(
            annotations["unique_member_count"].sum()
        ),
        "covered_raw_row_count": int(
            annotations["raw_member_count"].sum()
        ),
        "suggested_type_distribution": {
            str(key): int(value)
            for key, value in annotations[
                "suggested_type"
            ].value_counts().sort_index().items()
        },
        "review_status": {
            str(key): int(value)
            for key, value in annotations[
                "review_status"
            ].value_counts().sort_index().items()
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in dataset[
                "source"
            ].value_counts().sort_index().items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate group-level SMS type annotations."
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
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
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

    current_types = tuple(arguments.current_type)

    dataset = pd.read_csv(arguments.dataset)
    annotations = build_annotation_rows(
        dataset,
        target_label=arguments.label,
        current_types=current_types,
    )
    report = build_audit_report(
        dataset,
        annotations,
        target_label=arguments.label,
        current_types=current_types,
    )

    arguments.annotations.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    annotations.to_csv(
        arguments.annotations,
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
        "[Taxonomy audit] "
        f"label={report['target_label']} "
        f"raw_rows={report['raw_target_count']} "
        f"unique_rows={report['unique_target_count']} "
        f"groups={report['annotation_group_count']}"
    )
    print(
        f"[Taxonomy audit] annotations="
        f"{arguments.annotations}"
    )
    print(
        f"[Taxonomy audit] report="
        f"{arguments.report}"
    )


if __name__ == "__main__":
    main()