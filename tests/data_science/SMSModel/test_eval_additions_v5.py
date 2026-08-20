"""판정셋 추가분(#93) 무결성 테스트"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.analysis.text.preprocessing import URL_PATTERN, normalize_text
from data_science.SMSModel.data_quality import validate_sms_dataset
from data_science.SMSModel.evaluation.adoption import AdoptionCriteria
from data_science.SMSModel.merge_sms_diversity import (
    ADDITION_SOURCES,
    BASE_SOURCES,
)
from data_science.SMSModel.template_grouping import create_text_fingerprint
from data_science.SMSModel.train_sms import (
    ALLOWED_DATA_SOURCES,
    DATA_PATH,
    HOLDOUT_SOURCES,
    REAL_HOLDOUT_SOURCES,
    load_data,
    select_real_holdout,
)

EVAL_SOURCE = "real_holdout_v5"
ADDITIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data_science"
    / "Data"
    / "SMSData"
    / "sms_eval_additions_v5.csv"
)


@pytest.fixture(scope="module")
def additions() -> pd.DataFrame:
    """판정셋 추가분 CSV"""
    return pd.read_csv(ADDITIONS_PATH, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def judging_set() -> pd.DataFrame:
    """학습에 쓰이지 않는 실제 문자 판정셋(real_holdout)"""
    _, holdout = load_data(DATA_PATH)

    return select_real_holdout(holdout)


def test_source_is_registered_everywhere() -> None:
    """세 화이트리스트 중 하나라도 빠지면 병합이나 학습이 막힌다"""
    assert EVAL_SOURCE in ALLOWED_DATA_SOURCES
    assert EVAL_SOURCE in BASE_SOURCES
    assert EVAL_SOURCE in ADDITION_SOURCES


def test_source_routes_to_the_holdout() -> None:
    """평가 전용 source가 학습 pool로 새면 판정 자체가 무의미해진다

    #89의 real_collected_v4가 정확히 이 등록을 빠뜨려 164건 전부 학습에
    쓰였다. 학습에서 본 문자로 오탐률을 재면 실제보다 좋게 나온다.
    """
    assert EVAL_SOURCE in REAL_HOLDOUT_SOURCES
    assert EVAL_SOURCE in HOLDOUT_SOURCES


def test_additions_pass_dataset_validation(additions: pd.DataFrame) -> None:
    """label/type 조합과 source가 데이터셋 규약을 지켜야 한다"""
    validated = validate_sms_dataset(
        additions,
        allowed_sources=ADDITION_SOURCES,
    )

    assert len(validated) == len(additions)
    assert set(validated["source"]) == {EVAL_SOURCE}
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


def test_additions_reach_the_judging_set(
    additions: pd.DataFrame,
    judging_set: pd.DataFrame,
) -> None:
    """추가분이 학습 pool이 아니라 판정셋에 실제로 도착해야 한다

    source 화이트리스트 등록만으로는 부족하다. #89은 등록 자체를 빠뜨려
    164건이 통째로 학습에 쓰였으므로, 라우팅 결과를 직접 확인한다.
    """
    landed = int(judging_set["source"].eq(EVAL_SOURCE).sum())

    assert landed > 0
    assert landed <= len(additions)


def test_judging_set_clears_the_adoption_sample_floor(
    judging_set: pd.DataFrame,
) -> None:
    """판정셋 정상 표본이 채택 기준의 하한을 넘겨야 한다

    rule of three로 오탐 0건에서 상한 1%를 주장하려면 3/n <= 0.01, 즉 정상
    표본이 min_normal_samples 이상이어야 한다. 이 하한을 못 넘기면 어떤
    artifact도 INSUFFICIENT_EVIDENCE를 벗어날 수 없다.
    """
    normal_count = int(judging_set["label"].eq("normal").sum())

    assert normal_count >= AdoptionCriteria().min_normal_samples


def test_collection_covers_the_missing_types(
    additions: pd.DataFrame,
) -> None:
    """판정셋에 한 건도 없던 두 유형이 실제로 들어왔는지 확인

    인증알림은 #92가 좁힌 PERSONAL_INFO_PATTERN을, 공공기관알림은 v5에서
    오탐 점수 상위를 차지한 구간을 검증할 유일한 표본이다.
    """
    counts = additions["type"].value_counts()

    assert counts.get("정상인증알림", 0) >= 40
    assert counts.get("정상공공기관알림", 0) >= 25


def test_promotional_additions_carry_the_web_tag(
    additions: pd.DataFrame,
) -> None:
    """광고성 정상은 수신함 원문이어야 한다

    발송 문구 원본에는 [Web발신]가 없다. 태그가 빠진 표본이 섞이면 광고성
    정상의 태그 보유율이 떨어져 오탐이 오히려 나빠진다. 인증·공공기관처럼
    태그 없이 오는 유형이 많아 전체 비율로는 볼 수 없어 광고만 확인한다.
    """
    promotional = additions.loc[
        additions["type"].eq("정상광고프로모션"),
        "text",
    ]
    tagged = promotional.str.contains(r"\[Web발신\]", regex=True)

    assert tagged.mean() > 0.9
