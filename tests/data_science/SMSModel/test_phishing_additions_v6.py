"""피싱 학습셋 보강분(#102) 무결성 테스트"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.analysis.text.preprocessing import URL_PATTERN, normalize_text
from data_science.SMSModel.data_quality import validate_sms_dataset
from data_science.SMSModel.merge_sms_diversity import (
    ADDITION_SOURCES,
    BASE_SOURCES,
)
from data_science.SMSModel.template_grouping import create_text_fingerprint
from data_science.SMSModel.train_sms import (
    ALLOWED_DATA_SOURCES,
    DATA_PATH,
    HOLDOUT_SOURCES,
    load_data,
)

PHISHING_SOURCE = "real_phishing_v6"
ADDITIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data_science"
    / "Data"
    / "SMSData"
    / "sms_phishing_additions_v6.csv"
)

# 학습 표본이 부족해 base model이 놓치던 유형
TARGET_TYPES = ("지인가족사칭", "채용부업사기")


@pytest.fixture(scope="module")
def additions() -> pd.DataFrame:
    """피싱 보강분 CSV"""
    return pd.read_csv(ADDITIONS_PATH, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def training_pool() -> pd.DataFrame:
    """학습에 실제로 쓰이는 pool"""
    pool, _ = load_data(DATA_PATH)

    return pool


def test_source_is_registered_everywhere() -> None:
    """세 화이트리스트 중 하나라도 빠지면 병합이나 학습이 막힌다"""
    assert PHISHING_SOURCE in ALLOWED_DATA_SOURCES
    assert PHISHING_SOURCE in BASE_SOURCES
    assert PHISHING_SOURCE in ADDITION_SOURCES


def test_source_stays_out_of_the_holdout() -> None:
    """이 보강분은 학습 pool로 가야 한다

    #93의 real_holdout_v5와 반대다. 판정셋에 들어가면 391건 기준선이 바뀌어
    v5·v6·v7 판정 이력과 비교할 수 없게 된다.
    """
    assert PHISHING_SOURCE not in HOLDOUT_SOURCES


def test_additions_pass_dataset_validation(additions: pd.DataFrame) -> None:
    """label/type 조합과 source가 데이터셋 규약을 지켜야 한다"""
    validated = validate_sms_dataset(
        additions,
        allowed_sources=ADDITION_SOURCES,
    )

    assert len(validated) == len(additions)
    assert set(validated["source"]) == {PHISHING_SOURCE}
    assert set(validated["label"]) == {"phishing"}


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


def test_only_the_targeted_types_are_included(
    additions: pd.DataFrame,
) -> None:
    """이번 라운드는 두 유형만 다룬다

    투자리딩방사기 수집분은 SMS가 아니라 리딩방 대화 발췌라 보류했다. 판정셋의
    해당 유형은 유인 문자이므로 분포가 맞지 않는다.
    """
    assert set(additions["type"]) == set(TARGET_TYPES)


def test_targeted_types_grew_in_the_training_pool(
    training_pool: pd.DataFrame,
) -> None:
    """보강이 학습 pool에 실제로 도착했는지 확인

    source 등록만으로는 부족하다. #89은 등록을 빠뜨려 164건이 통째로 학습에
    쓰였고, 이번에는 반대로 학습에 도착해야 한다.
    """
    phishing = training_pool[training_pool["label"].eq("phishing")]
    counts = phishing["type"].value_counts()

    # 보강 전 지인가족사칭 6건, 채용부업사기 2건이었다
    assert counts.get("지인가족사칭", 0) > 6
    assert counts.get("채용부업사기", 0) > 2
