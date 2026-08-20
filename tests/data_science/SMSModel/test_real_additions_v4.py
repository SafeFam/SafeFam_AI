"""실수집 문자 추가분(#89) 무결성 테스트"""

from __future__ import annotations

import pathlib
import re
from pathlib import Path

import pandas as pd
import pytest

from app.analysis.text.preprocessing import URL_PATTERN, normalize_text
from data_science.SMSModel.data_quality import validate_sms_dataset
from data_science.SMSModel.merge_sms_diversity import (
    ADDITION_SOURCES,
    BASE_SOURCES,
)
from data_science.SMSModel import run_stacking_training as training
from data_science.SMSModel.template_grouping import create_text_fingerprint
from data_science.SMSModel.train_sms import (
    ALLOWED_DATA_SOURCES,
    SPLIT_MANIFEST_PATH,
)

REAL_SOURCE = "real_collected_v4"
ADDITIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data_science"
    / "Data"
    / "SMSData"
    / "sms_real_additions_v4.csv"
)


@pytest.fixture(scope="module")
def additions() -> pd.DataFrame:
    """수집 추가분 CSV"""
    return pd.read_csv(ADDITIONS_PATH, encoding="utf-8-sig")


def test_source_is_registered_everywhere() -> None:
    """세 화이트리스트 중 하나라도 빠지면 병합이나 학습이 막힌다"""
    assert REAL_SOURCE in ALLOWED_DATA_SOURCES
    assert REAL_SOURCE in BASE_SOURCES
    assert REAL_SOURCE in ADDITION_SOURCES


def test_split_manifest_matches_the_training_guard() -> None:
    """학습 스크립트가 요구하는 manifest와 실제 경로가 어긋나면 안 된다

    데이터가 바뀔 때마다 manifest를 새로 만드는데, 학습 쪽 검사에 버전이
    문자열로 박혀 있어 한쪽만 갱신되면 학습이 통째로 막힌다.
    """
    source = pathlib.Path(training.__file__).read_text(encoding="utf-8")
    required = re.search(r'SPLIT_MANIFEST_PATH\.name != "([^"]+)"', source)

    assert required is not None
    assert SPLIT_MANIFEST_PATH.name == required.group(1)
    assert SPLIT_MANIFEST_PATH.is_file()


def test_additions_pass_dataset_validation(additions: pd.DataFrame) -> None:
    """label/type 조합과 source가 데이터셋 규약을 지켜야 한다"""
    validated = validate_sms_dataset(
        additions,
        allowed_sources=ADDITION_SOURCES,
    )

    assert len(validated) == len(additions)
    assert set(validated["source"]) == {REAL_SOURCE}
    assert set(validated["label"]) == {"normal"}


def test_every_addition_is_reviewed(additions: pd.DataFrame) -> None:
    """검수 표시가 없으면 병합 단계에서 거부된다"""
    assert set(additions["review_status"]) == {"APPROVED"}
    assert additions["reviewer"].fillna("").str.strip().ne("").all()
    assert additions["review_note"].fillna("").str.strip().ne("").all()


def test_additions_have_no_internal_duplicates(
    additions: pd.DataFrame,
) -> None:
    """마스킹 후 같아지는 행은 한 건 값어치다"""
    fingerprints = additions["text"].map(
        lambda text: create_text_fingerprint(normalize_text(text))
    )

    assert not fingerprints.duplicated().any()


def test_has_url_matches_the_text(additions: pd.DataFrame) -> None:
    """has_url 컬럼이 본문과 어긋나면 구조 특징이 뒤틀린다"""
    expected = additions["text"].map(lambda t: bool(URL_PATTERN.search(t)))

    assert additions["has_url"].astype(bool).equals(expected)


def test_collection_covers_the_targeted_gaps(
    additions: pd.DataFrame,
) -> None:
    """#89가 지목한 세 유형이 실제로 들어왔는지 확인

    광고성 정상은 오탐이 몰린 구간이고, 공공기관·인증 두 유형은 평가셋에
    한 건도 없어 성능 측정 자체가 불가능했던 구간이다.
    """
    counts = additions["type"].value_counts()

    assert counts.get("정상광고프로모션", 0) >= 40
    assert counts.get("정상공공기관알림", 0) > 0
    assert counts.get("정상인증알림", 0) > 0


def test_additions_carry_the_web_tag(additions: pd.DataFrame) -> None:
    """수신함 원문에는 [Web발신]가 붙는다

    발송 문구 원본에는 이 태그가 없다. 태그가 빠진 표본이 섞이면 광고성
    정상의 태그 보유율이 떨어져 오탐이 오히려 나빠진다.
    """
    tagged = additions["text"].str.contains(r"\[Web발신\]", regex=True)

    assert tagged.mean() > 0.95
