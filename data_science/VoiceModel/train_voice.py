"""
SafeFam — 음성(STT) 피싱 탐지 베이즈 분류기 학습 스크립트
train_voice.py
===========================================================
Author : Backend B (기범)
Dataset: metadata_clean.csv

SMS 대비 주요 차이점:
  - 입력: 구어체 STT 텍스트 (text_clean 컬럼, 전처리 완료본)
  - split: 메타데이터 split 컬럼을 신뢰하지 않고 parent_call_id 기준으로
    직접 재분할 (아래 [데이터 누수 수정] 참고)
  - 튜닝: validation으로 alpha/threshold 탐색, test는 최종 평가 전용
  - 벡터화: char n-gram vs word n-gram 비교 후 최적 선택
  - 클래스 불균형: normal < phishing → class_prior 보정

[데이터 누수 수정 — 2026-07]
  검증 셀 1/2(VoiceData.ipynb)에서 parent_call_id 32건, text_clean 23~34건이
  train/test에 동시 존재하는 누수 발견. 원인 두 가지:
    1) 증강(is_augmented=True) 600건이 parent_call_id로 그룹화되지 않은 채
       개별 행 단위로 split이 배정됨 (동일 통화의 세그먼트가 서로 다른 split에 흩어짐)
    2) 보이스피싱 시나리오가 스크립트 템플릿을 재사용해, 서로 다른 통화인데도
       text_clean이 완전히 동일한 경우가 있어 parent_call_id만 묶어도 텍스트 누수가 남음
  → 메타데이터의 split 컬럼을 사용하지 않고, parent_call_id와 text_clean 두 기준을
    Union-Find로 묶어 그룹 단위 StratifiedShuffleSplit 2단계 재분할(_leak_free_split)을
    직접 수행. 두 키 중 하나라도 같으면 반드시 같은 split에 배정되므로 두 종류의
    누수가 모두 구조적으로 불가능해진다.

[완전 신규 시나리오 홀드아웃 — 2026-07]
  source_dataset == 'synthetic_new_holdout' / 'synthetic_fp_stress' 행은
  leak-free 재분할 풀에서 아예 제외하고 df_holdout으로 따로 뺀다. 학습에
  전혀 관여하지 않은 완전 unseen 데이터로 일반화 성능(recall)과 오탐률(FP)을
  동시에 검증하기 위함 — evaluate_new_holdout() 참고.

[구조적 위험 신호 하이브리드 보정 — 2026-07]
  NB가 학습 못 한 어휘(신규 사기 유형) 때문에 확신 없이 낮은 점수를 매겨도,
  계좌·개인정보·긴급성 등 위험 신호가 뚜렷하면 risk_score 하한선을 강제로
  끌어올려 최소 LLM 에스컬레이션(MEDIUM) 구간까지는 보장한다.
  단독 신호 하나로 정상 업무 전화까지 과탐지하지 않도록 가중치를 실측
  튜닝했다 — _apply_risk_floor() 참고.

탐지 파이프라인 내 위치:
  음성 통화 → STT 변환 → text_clean 정규화 → 이 모델 → risk_score
  (SMS 파이프라인과 별도 모델로 운영)

저장 파일:
  voice_model_artifact.pkl  — model + threshold + classes + vectorizer_type
  voice_vectorizer.pkl      — 학습된 벡터라이저
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import ComplementNB, MultinomialNB

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 경로 CONFIG — 환경에 따라 이 블록만 수정
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = Path("../Data/CallData/metadata_clean.csv")  # generate_voice_data.py 출력본
MODEL_PATH = Path("voice_model_artifact.pkl")
VECTORIZER_PATH = Path("voice_vectorizer.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 학습 CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.15  # parent_call_id 기준 재분할 시 test 비중 (SMS와 동일 기준)
VAL_SIZE = 0.15  # parent_call_id 기준 재분할 시 validation 비중
TARGET_PHISHING_RECALL = 0.96  # 피싱 미탐(FN) > 정상 오탐(FP)

# risk_level 구간 (SMS와 동일 기준)
RISK_HIGH_THRESHOLD = 70  # HIGH   : 70점 이상
RISK_MEDIUM_THRESHOLD = 40  # MEDIUM : 40~69점 (LLM 에스컬레이션 대상)
# LOW    : 40점 미만

# 격자 탐색 범위
ALPHA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
THRESHOLD_GRID = np.round(np.arange(0.30, 0.75, 0.05), 2)

# 벡터화 후보 — char n-gram vs word n-gram 비교
# STT 구어체 특성상 word n-gram이 char보다 나을 수 있어 두 가지 모두 탐색
VECTORIZER_CANDIDATES: dict[str, CountVectorizer] = {
    "char_ngram": CountVectorizer(
        analyzer="char_wb",  # 단어 경계 존중 문자 n-gram (SMS에서 검증됨)
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
        max_features=8_000,
    ),
    "word_ngram": CountVectorizer(
        analyzer="word",  # 단어 단위 토큰화 (구어체에 더 직관적)
        ngram_range=(1, 2),  # unigram + bigram
        min_df=2,
        max_df=0.95,
        max_features=8_000,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 구조적 위험 신호 피처 — 정규식 정의 및 추출
# ─────────────────────────────────────────────────────────────────────────────
# [2026-07 보강] 신규 holdout(택배/환급금/알바 사칭) 테스트에서 기존 패턴이
# 대출사기형/수사기관 사칭형 어휘에만 좁게 맞춰져 있던 게 드러나 일반화된 카테고리로 확장.

_RE_URGENCY = re.compile(
    r"지금\s*바로|즉시|긴급|당장|오늘\s*(안에|중으로)|시간이\s*없|빨리\s*확인"
)
_RE_AUTHORITY = re.compile(
    r"검찰|경찰|금감원|금융감독원|국세청|법원|수사관|형사|공단|세관"
)
_RE_MONEY = re.compile(r"대출|이자|원금|계좌|송금|이체|입금|출금|돈|환급|관세")
_RE_PERSONAL = re.compile(
    r"주민\s*번호|계좌\s*번호|비밀\s*번호|카드\s*번호|개인\s*정보|본인\s*인증"
)
_RE_THREAT = re.compile(
    r"체포|구속|압수|수색|처벌|벌금|기소|고소|고발|반송\s*처리|제외"
)
_RE_SAFE_ACCT = re.compile(
    r"안전\s*계좌|보호\s*계좌|임시\s*계좌|자금\s*보호|보증금|예치금"
)
_RE_INSTALL = re.compile(r"앱\s*설치|다운로드|원격|팀뷰어|애니데스크|링크")
_RE_ADVANCE = re.compile(
    r"활동비|수수료|보증금|예치금|선\s*입금|먼저\s*입금"
)  # 선입금 요구형 공통 패턴
_RE_LONG_TEXT = re.compile(r".{200,}")  # 200자 초과 — 보이스피싱 설명 특성

# 메신저피싱(가족·지인 사칭) 특유의 '기기 이상 핑계 + 대리 연락' 패턴.
# "폰 고장/액정 깨짐/번호 바뀜"을 대며 본인 확인을 피하고, "지금 아니면 안 된다"는
# 식으로 되묻지 못하게 만드는 게 이 유형의 핵심 수법이라 별도 피처화.
_RE_IMPERSONATION_EXCUSE = re.compile(
    r"폰\s*(고장|액정|깨져)|다른\s*사람\s*폰|번호가?\s*바뀌|통화\s*(가|는)\s*힘들"
)

# _extract_struct_features()가 반환하는 배열의 열 순서와 정확히 일치해야 함.
# 피처 중요도 시각화(VoiceData.ipynb)에서 이 리스트를 그대로 import해서 쓴다 —
# 노트북에 이름을 따로 하드코딩하면 피처 추가/변경 시 어긋나기 쉽기 때문.
STRUCT_FEATURE_NAMES = [
    "has_urgency",
    "has_authority",
    "has_money",
    "has_personal",
    "has_threat",
    "has_safe_acct",
    "has_install",
    "is_long_text",
    "has_advance_fee",
    "has_impersonation_excuse",
]


def _extract_struct_features(texts: pd.Series) -> np.ndarray:
    """
    STT 구어체 특화 구조적 피처 10개 추출.

    피처 목록:
      0: has_urgency               — 긴급성 유도 표현
      1: has_authority             — 수사/금융기관 사칭 키워드
      2: has_money                 — 금전 관련 키워드
      3: has_personal              — 개인정보 요구 표현
      4: has_threat                — 법적 위협/불이익 표현
      5: has_safe_acct             — 안전계좌 유도 표현
      6: has_install                — 앱 설치/링크 유도 표현
      7: is_long_text              — 200자 초과
      8: has_advance_fee           — 선입금/보증금/수수료 요구
      9: has_impersonation_excuse  — 기기 이상 핑계 + 대리 연락(메신저피싱 특유 패턴)
    """
    return np.column_stack(
        [
            texts.str.contains(_RE_URGENCY, regex=True).astype(int).values,
            texts.str.contains(_RE_AUTHORITY, regex=True).astype(int).values,
            texts.str.contains(_RE_MONEY, regex=True).astype(int).values,
            texts.str.contains(_RE_PERSONAL, regex=True).astype(int).values,
            texts.str.contains(_RE_THREAT, regex=True).astype(int).values,
            texts.str.contains(_RE_SAFE_ACCT, regex=True).astype(int).values,
            texts.str.contains(_RE_INSTALL, regex=True).astype(int).values,
            texts.str.contains(_RE_LONG_TEXT, regex=True).astype(int).values,
            texts.str.contains(_RE_ADVANCE, regex=True).astype(int).values,
            texts.str.contains(_RE_IMPERSONATION_EXCUSE, regex=True).astype(int).values,
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 하이브리드 보정 — NB가 놓쳐도 최소 위험 신호가 있으면 LLM 에스컬레이션 구간까지는 올림
# ─────────────────────────────────────────────────────────────────────────────
# [2026-07 재조정] 오탐 스트레스 테스트 결과 '계좌/카드번호/본인인증'만으로
# floor가 발동해 정상 은행 업무 전화까지 MEDIUM 이상으로 밀려 올라가는 게
# 확인됨 → has_money/has_personal 하향 조정.

# 각 구조 피처의 위험 가중치. 개인정보/설치유도/선입금 요구/대리연락 핑계처럼
# 정상 통화에서 거의 안 나오는 신호는 높게, 금전 언급처럼 정상 은행 안내
# 전화에도 흔한 신호는 낮게 잡아 오탐(FP) 증가를 최소화한다.
_STRUCT_WEIGHTS = np.array(
    [
        1,  # has_urgency
        1,  # has_authority
        0.5,  # has_money                 (계좌/환급 등 정상 은행 업무에도 매우 흔함)
        1,  # has_personal              (카드번호/본인인증만으로는 약한 신호)
        2,  # has_threat
        2,  # has_safe_acct
        2,  # has_install
        0.5,  # is_long_text
        2,  # has_advance_fee
        4,  # has_impersonation_excuse  (메신저사칭형 recall 강화용)
    ]
)

# 정상 업무 맥락 신호 — 고객센터/영업일/재발급 같은 표현이 있으면 사기 가능성을
# 낮게 보고 가중합에서 차감한다. 실제 사기 스크립트는 이런 '내부 업무 용어'를
# 거의 안 쓰고, 대신 절차를 급하게 스킵시키려는 표현을 쓴다.
# 주의: "정산"만 단독으로 걸면 "첫 정산 때 수익이랑 같이 돌려드립니다" 같은
# 사기 미끼 문구까지 정상으로 오인하므로, "정산 완료/내역/이력"처럼 이미
# 끝난 업무를 가리킬 때만 정상 신호로 인정한다.
_RE_LEGIT_CONTEXT = re.compile(
    r"고객센터|영업일|재발급|정산\s*(완료|내역|이력)|검수|승인\s*거부|해지|만료\s*예정"
)

# 가중합 구간별 최소 risk_score. NB 확률이 아무리 낮아도 이 값 밑으로는
# 안 떨어지도록 max()로 강제한다.
_RISK_FLOOR_TABLE = [
    (5, 70),  # 위험 신호 가중합 5 이상   → 최소 HIGH
    (3.5, 55),  # 3.5 이상                 → 최소 MEDIUM 상단
    (2, 40),  # 2 이상                   → 최소 MEDIUM 진입 (LLM 에스컬레이션 보장)
]


def _apply_risk_floor(risk_score: int, struct_row: np.ndarray, text: str) -> int:
    """
    구조적 위험 신호 가중합에 따라 risk_score의 하한선을 강제 적용.
    NB가 신규/미학습 어휘 때문에 확신 없이 LOW로 떨어뜨려도,
    위험 신호가 뚜렷하면 최소 MEDIUM 이상까지는 끌어올려 LLM 에스컬레이션을 보장한다.
    단, '고객센터/영업일/재발급' 같은 정상 업무 문맥이 있으면 감점해서
    정상 안내 전화가 과도하게 위험군으로 분류되는 걸 완화한다.
    """
    weighted_sum = float(np.dot(struct_row, _STRUCT_WEIGHTS))
    if _RE_LEGIT_CONTEXT.search(text):
        weighted_sum -= 1.5
    for min_weight, floor_score in _RISK_FLOOR_TABLE:
        if weighted_sum >= min_weight:
            return max(risk_score, floor_score)
    return risk_score


def _map_risk_level(risk_score: int) -> str:
    """
    risk_score → risk_level 매핑 (SMS 모델과 동일 기준).
    HIGH   ≥ 70점 : 즉시 위험 판단
    MEDIUM 40~69점: LLM 에스컬레이션 대상
    LOW    < 40점 : 정상 범주
    """
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    if risk_score >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────


def _leak_free_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    parent_call_id와 text_clean 두 기준 모두에서 누수가 없도록 Union-Find로 그룹을
    묶은 뒤, 그룹 단위 StratifiedShuffleSplit 2단계로 train/val/test를 분리한다.

    메타데이터의 split 컬럼은 사용하지 않는다:
      - parent_call_id 누락: 증강(is_augmented=True) 데이터가 통화 단위가 아닌
        행 단위로 개별 split 배정되어, 같은 통화의 세그먼트가 train/test에 흩어짐
        (검증 셀 1에서 parent_call_id 32건 중복 확인).
      - text_clean 누락: 보이스피싱 시나리오가 스크립트 템플릿을 재사용해 서로 다른
        통화(parent_call_id)인데도 text_clean이 완전히 동일한 경우가 있어,
        parent_call_id만으로 묶어도 텍스트 누수가 남음
        (검증 셀 2에서 34건 중복 확인 — 템플릿 스크립트 재사용).
    → 두 키 중 하나라도 같으면 같은 그룹으로 묶어 반드시 같은 split에 배정한다
      (Union-Find). 그룹 단위 분할이므로 SMS의 split_data()와 달리 개별 행이 아닌
      그룹에 대해 StratifiedShuffleSplit 2단계를 적용한다.
    """
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    df = df.reset_index(drop=True)
    for key in ("parent_call_id", "text_clean"):
        for _, idxs in df.groupby(key).groups.items():
            idxs = list(idxs)
            for i in idxs[1:]:
                union(idxs[0], i)

    df = df.assign(_group=[find(i) for i in range(n)])
    groups = df.groupby("_group")["label"].first().reset_index()

    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trainval_idx, test_idx = next(sss_test.split(groups, groups["label"]))
    trainval_groups = groups.iloc[trainval_idx]
    test_ids = set(groups.iloc[test_idx]["_group"])

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(sss_val.split(trainval_groups, trainval_groups["label"]))
    train_ids = set(trainval_groups.iloc[train_idx]["_group"])
    val_ids = set(trainval_groups.iloc[val_idx]["_group"])

    df_train = (
        df[df["_group"].isin(train_ids)].drop(columns="_group").reset_index(drop=True)
    )
    df_val = (
        df[df["_group"].isin(val_ids)].drop(columns="_group").reset_index(drop=True)
    )
    df_test = (
        df[df["_group"].isin(test_ids)].drop(columns="_group").reset_index(drop=True)
    )

    return df_train, df_val, df_test


def load_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    metadata_clean.csv 로드 후 parent_call_id 기준으로 leak-free train/val/test 분리 반환.

    source_dataset이 'synthetic_new_holdout' 또는 'synthetic_fp_stress'인 행은
    leak-free 재분할 풀에서 완전히 제외하고 df_holdout으로 따로 반환한다.
    학습/튜닝/기존 test 평가에 일절 관여하지 않는 순수 외부 검증용 데이터로
    유지하기 위함 (완전 신규 시나리오 recall + 오탐 스트레스 테스트 겸용).

    Returns:
        (df_train, df_val, df_test, df_holdout)
    """
    df = pd.read_csv(path)

    required = {"text_clean", "label", "parent_call_id"}
    if missing := required - set(df.columns):
        raise ValueError(f"누락 컬럼: {missing}")
    if df[list(required)].isnull().any().any():
        raise ValueError("text_clean / label / parent_call_id 중 결측치 존재")
    if not {"phishing", "normal"}.issuperset(set(df["label"].unique())):
        raise ValueError(f"예상치 못한 label 값: {df['label'].unique()}")

    is_new_holdout = df["source_dataset"].isin(
        ["synthetic_new_holdout", "synthetic_fp_stress"]
    )
    df_holdout = df[is_new_holdout].reset_index(drop=True)
    df_pool = df[~is_new_holdout].reset_index(drop=True)

    df_train, df_val, df_test = _leak_free_split(df_pool)

    def _fmt(d: pd.DataFrame) -> str:
        vc = d["label"].value_counts()
        return (
            f"phishing={vc.get('phishing', 0)} "
            f"({vc.get('phishing', 0) / len(d):.1%}) | "
            f"normal={vc.get('normal', 0)} "
            f"({vc.get('normal', 0) / len(d):.1%})"
        )

    print(
        f"[Load] 전체 {len(df)}건 (신규 holdout {len(df_holdout)}건 분리) | "
        f"재분할 풀 {len(df_pool)}건"
    )
    print(f"  Train   : {len(df_train)}건 — {_fmt(df_train)}")
    print(f"  Val     : {len(df_val)}건  — {_fmt(df_val)}")
    print(f"  Test    : {len(df_test)}건 — {_fmt(df_test)}")
    print(
        f"  Holdout : {len(df_holdout)}건 — {_fmt(df_holdout)}  (학습에 전혀 사용 안 됨)"
    )

    return df_train, df_val, df_test, df_holdout


# ─────────────────────────────────────────────────────────────────────────────
# 피처 행렬 구성
# ─────────────────────────────────────────────────────────────────────────────


def build_feature_matrix(
    vectorizer: CountVectorizer,
    texts: pd.Series,
    struct: np.ndarray,
    *,
    fit: bool,
):
    """
    텍스트 피처(CountVectorizer) + 구조적 피처(_extract_struct_features) sparse hstack 결합.

    Args:
        fit: True → fit_transform (학습 전용), False → transform only (누수 방지)
    """
    X_text = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    X_struct = csr_matrix(struct)
    return hstack([X_text, X_struct])


# ─────────────────────────────────────────────────────────────────────────────
# 학습 및 튜닝
# ─────────────────────────────────────────────────────────────────────────────


def _phishing_idx(model) -> int:
    """model.classes_ 에서 'phishing' 인덱스 반환."""
    return list(model.classes_).index("phishing")


def _compute_recalls(
    model,
    X_val,
    y_val: pd.Series,
    threshold: float,
) -> tuple[float, float]:
    """threshold 적용 후 Recall(phishing), Recall(normal) 반환."""
    y_prob = model.predict_proba(X_val)[:, _phishing_idx(model)]
    y_pred = np.where(y_prob >= threshold, "phishing", "normal")
    cm = confusion_matrix(y_val, y_pred, labels=["normal", "phishing"])

    rec_p = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    rec_n = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    return rec_p, rec_n


def train_and_tune(
    X_train,
    y_train: pd.Series,
    X_val,
    y_val: pd.Series,
    vec_type: str,
) -> dict:
    """
    [격자 탐색] 모델(ComplementNB / MultinomialNB) × alpha × threshold 전체 탐색.
    튜닝 기준: validation 세트 사용 (test는 최종 평가 전용으로 봉인)

    선택 기준:
      1순위) Recall(phishing) >= TARGET 조건 하 Recall(normal) 최대
      2순위) 목표 미달 시 Recall(phishing) 최대 (fallback)

    Args:
        vec_type: 현재 탐색 중인 벡터화 방식 이름 (로그 출력용)

    Returns:
        dict: model, model_name, alpha, threshold, recall_phishing, recall_normal
    """
    phishing_ratio = (y_train == "phishing").mean()
    normal_ratio = 1 - phishing_ratio

    best: dict = {
        "model": None,
        "model_name": "",
        "alpha": None,
        "threshold": None,
        "recall_phishing": 0.0,
        "recall_normal": 0.0,
    }

    header = (
        f"{'모델':>15} | {'alpha':>5} | {'thresh':>6} "
        f"| {'rec_phish':>10} | {'rec_normal':>10}"
    )
    print(f"\n{'=' * 64}")
    print(f"[ 격자 탐색 — {vec_type} ]")
    print(f"{'=' * 64}")
    print(header)
    print("-" * len(header))

    for model_cls in [ComplementNB, MultinomialNB]:
        for alpha in ALPHA_GRID:
            # CalibratedClassifierCV: 확률이 0/1 양극단으로 몰리는 문제 해결
            # isotonic: 비선형 보정 — 작은 데이터셋에서 sigmoid보다 안정적
            base = (
                MultinomialNB(
                    alpha=alpha,
                    class_prior=[normal_ratio, phishing_ratio],  # [normal, phishing]
                )
                if model_cls is MultinomialNB
                else ComplementNB(alpha=alpha)
            )
            model = CalibratedClassifierCV(
                estimator=base,
                method="isotonic",
                cv=5,
            )
            model.fit(X_train, y_train)

            for threshold in THRESHOLD_GRID:
                rec_p, rec_n = _compute_recalls(model, X_val, y_val, threshold)

                print(
                    f"{model_cls.__name__:>15} | {alpha:>5} | {threshold:>6.2f} "
                    f"| {rec_p:>10.4f} | {rec_n:>10.4f}"
                )

                # 목표 달성 + 현재 최고 normal recall 갱신
                if (
                    rec_p >= TARGET_PHISHING_RECALL
                    and rec_n > best["recall_normal"]
                    or best["model"] is None
                    and rec_p > best["recall_phishing"]
                ):
                    best.update(
                        {
                            "model": model,
                            "model_name": model_cls.__name__,
                            "alpha": alpha,
                            "threshold": threshold,
                            "recall_phishing": rec_p,
                            "recall_normal": rec_n,
                        }
                    )

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 벡터화 후보 비교
# ─────────────────────────────────────────────────────────────────────────────


def compare_vectorizers(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[str, CountVectorizer, dict]:
    """
    char n-gram vs word n-gram 두 가지 벡터화 방식을 각각 격자 탐색하고
    validation Recall(phishing) 기준 최적 조합을 반환.

    Returns:
        (best_vec_type, best_vectorizer, best_result_dict)
    """
    struct_train = _extract_struct_features(df_train["text_clean"])
    struct_val = _extract_struct_features(df_val["text_clean"])

    comparison: dict[str, dict] = {}

    for vec_type, vectorizer in VECTORIZER_CANDIDATES.items():
        # train에만 fit — val은 transform only (데이터 누수 방지)
        X_train = build_feature_matrix(
            vectorizer, df_train["text_clean"], struct_train, fit=True
        )
        X_val = build_feature_matrix(
            vectorizer, df_val["text_clean"], struct_val, fit=False
        )

        result = train_and_tune(
            X_train,
            df_train["label"],
            X_val,
            df_val["label"],
            vec_type,
        )
        result["vectorizer"] = vectorizer
        result["vectorizer_type"] = vec_type
        comparison[vec_type] = result

    # 비교 요약 출력
    print(f"\n{'=' * 64}")
    print("[ 벡터화 방식 비교 요약 ]")
    print(f"{'=' * 64}")
    print(
        f"{'방식':>12} | {'모델':>15} | {'alpha':>5} | {'thresh':>6} | {'rec_phish':>10} | {'rec_normal':>10}"
    )
    print("-" * 70)
    for vec_type, result in comparison.items():
        if result["model"] is not None:
            print(
                f"{vec_type:>12} | {result['model_name']:>15} | "
                f"{result['alpha']:>5} | {result['threshold']:>6.2f} | "
                f"{result['recall_phishing']:>10.4f} | {result['recall_normal']:>10.4f}"
            )

    # Recall(phishing) >= TARGET 조건 하 Recall(normal) 최대인 벡터화 선택
    valid = {
        k: v
        for k, v in comparison.items()
        if v["model"] is not None and v["recall_phishing"] >= TARGET_PHISHING_RECALL
    }

    if valid:
        best_type = max(valid, key=lambda k: valid[k]["recall_normal"])
    else:
        # 목표 미달 시 Recall(phishing) 최대
        best_type = max(
            comparison,
            key=lambda k: (
                comparison[k]["recall_phishing"] if comparison[k]["model"] else 0
            ),
        )
        print("[WARNING] 어떤 조합도 Recall(phishing) 목표를 달성하지 못했습니다.")

    best_result = comparison[best_type]
    print(f"\n★ 최적 벡터화: {best_type}")
    print(
        f"  모델={best_result['model_name']} | alpha={best_result['alpha']} "
        f"| threshold={best_result['threshold']}"
    )
    print(
        f"  Recall(phishing)={best_result['recall_phishing']:.4f} | "
        f"Recall(normal)={best_result['recall_normal']:.4f}"
    )

    return best_type, best_result["vectorizer"], best_result


# ─────────────────────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model,
    threshold: float,
    X_test,
    y_test: pd.Series,
) -> None:
    """
    test 세트로 최종 평가.
    validation으로 튜닝한 결과가 test에서도 유지되는지 검증.
    """
    y_prob = model.predict_proba(X_test)[:, _phishing_idx(model)]
    y_pred = np.where(y_prob >= threshold, "phishing", "normal")

    print(f"\n{'=' * 64}")
    print(f"[ 최종 평가 (test 세트) — threshold={threshold} ]")
    print(f"{'=' * 64}")
    print(classification_report(y_test, y_pred, target_names=["normal", "phishing"]))

    cm = confusion_matrix(y_test, y_pred, labels=["normal", "phishing"])
    print(
        pd.DataFrame(
            cm,
            index=["실제 normal", "실제 phishing"],
            columns=["예측 normal", "예측 phishing"],
        )
    )

    rec_p = cm[1, 1] / cm[1].sum()
    rec_n = cm[0, 0] / cm[0].sum()
    print(f"\n★ Recall(phishing): {rec_p:.4f}  |  Recall(normal): {rec_n:.4f}")

    if rec_p < TARGET_PHISHING_RECALL:
        print(
            f"[WARNING] Recall(phishing) {rec_p:.4f} < 목표 {TARGET_PHISHING_RECALL:.2f} — "
            "데이터 다양성 추가 또는 하이퍼파라미터 재조정 권장"
        )


def evaluate_new_holdout(
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    df_holdout: pd.DataFrame,
) -> None:
    """
    학습에 전혀 관여하지 않은 완전 신규 시나리오(synthetic_new_holdout,
    synthetic_fp_stress)로 최종 일반화 성능 + 오탐률을 검증.
    구조 피처 보정(_apply_risk_floor)을 반영한 risk_score 기준으로 판정하며,
    call_type(신규 유형)별 recall / 오탐률도 함께 출력한다.
    """
    if df_holdout.empty:
        print("\n[SKIP] 신규 holdout 데이터 없음")
        return

    struct = _extract_struct_features(df_holdout["text_clean"])
    X = build_feature_matrix(vectorizer, df_holdout["text_clean"], struct, fit=False)

    y_prob = model.predict_proba(X)[:, _phishing_idx(model)]
    raw_scores = (y_prob * 100).astype(int)

    # 구조 피처 보정 적용 — NB가 놓쳐도 위험 신호가 뚜렷하면 최소 점수 하한 적용
    boosted_scores = np.array(
        [
            _apply_risk_floor(s, struct[i], df_holdout["text_clean"].iloc[i])
            for i, s in enumerate(raw_scores)
        ]
    )
    # threshold * 100의 부동소수점 오차(예: 0.55*100 → 55.00000000000001) 때문에
    # 경계값이 누락되는 걸 방지하기 위해 정수 스케일로 반올림 후 비교
    threshold_pct = round(threshold * 100)
    y_pred = np.where(boosted_scores >= threshold_pct, "phishing", "normal")

    print(f"\n{'=' * 64}")
    print(
        f"[ 완전 신규 시나리오 holdout 평가 — {len(df_holdout)}건 ] (구조 피처 보정 적용)"
    )
    print(f"{'=' * 64}")
    print(
        classification_report(
            df_holdout["label"], y_pred, target_names=["normal", "phishing"]
        )
    )

    cm = confusion_matrix(df_holdout["label"], y_pred, labels=["normal", "phishing"])
    print(
        pd.DataFrame(
            cm,
            index=["실제 normal", "실제 phishing"],
            columns=["예측 normal", "예측 phishing"],
        )
    )

    rec_p = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    rec_n = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    print(f"\n★ Recall(phishing): {rec_p:.4f}  |  Recall(normal): {rec_n:.4f}")

    df_h = df_holdout.copy()
    df_h["pred"] = y_pred
    df_h["raw_score"] = raw_scores  # NB 원본 점수 (보정 전, 비교용)
    df_h["risk_score"] = boosted_scores  # 보정 후 최종 점수

    # ── 유형(call_type)별 recall — 어떤 신규 시나리오가 취약한지 확인 ──
    phishing_df = df_h[df_h["label"] == "phishing"]
    if not phishing_df.empty:
        print("\n[유형별 Recall(phishing)] — 낮은 순")
        recall_by_type = (
            phishing_df.groupby("call_type")
            .apply(lambda g: (g["pred"] == "phishing").mean())
            .sort_values()
        )
        print(recall_by_type)

        fn = phishing_df[phishing_df["pred"] == "normal"].copy()
        if not fn.empty:
            below_low = (fn["risk_score"] < RISK_MEDIUM_THRESHOLD).sum()
            print(f"\n[놓친 phishing {len(fn)}건 중 risk_score 분포 (보정 후)]")
            print(f"  LOW(<40, 완전히 놓침)      : {below_low}건")
            print(f"  MEDIUM(40~69, LLM 구제 가능) : {len(fn) - below_low}건")
            print(
                fn[["call_type", "raw_score", "risk_score", "text_clean"]]
                .sort_values("risk_score", ascending=False)
                .head(10)
                .to_string()
            )

    # ── normal 오탐 분석: HIGH로 잘못 확정된 건지, MEDIUM(LLM 검토용)인지 구분 ──
    normal_df = df_h[df_h["label"] == "normal"].copy()
    if not normal_df.empty:
        normal_df["risk_level"] = normal_df["risk_score"].apply(_map_risk_level)
        fp = normal_df[normal_df["pred"] == "phishing"]

        print(
            f"\n[정상 통화 오탐(FP) {len(fp)}건 / 전체 정상 {len(normal_df)}건 — risk_level 분포]"
        )
        print(fp["risk_level"].value_counts())

        print("\n[유형(call_type)별 오탐률]")
        fpr_by_type = (
            normal_df.groupby("call_type")
            .apply(lambda g: (g["pred"] == "phishing").mean())
            .sort_values(ascending=False)
        )
        print(fpr_by_type)

        high_fp = fp[fp["risk_level"] == "HIGH"]
        if not high_fp.empty:
            print(
                "\n[HIGH로 잘못 확정된 정상 통화 — LLM 안전망도 못 거침, 최우선 수정 대상]"
            )
            print(
                high_fp[["call_type", "risk_score", "text_clean"]]
                .sort_values("risk_score", ascending=False)
                .head(10)
                .to_string()
            )


def diagnose_low_phishing_full_dataset(
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    data_path: Path,
) -> None:
    """
    전체 데이터셋(train+val+test+holdout 통합) 기준으로 risk_level=LOW인 phishing이
    어느 source_dataset/call_type에서 나오는지 진단.
    train 데이터가 포함되므로 '일반화 테스트'가 아니라 사후 감사(audit) 목적임에 주의.
    """
    df = pd.read_csv(data_path)
    struct = _extract_struct_features(df["text_clean"])
    X = build_feature_matrix(vectorizer, df["text_clean"], struct, fit=False)

    y_prob = model.predict_proba(X)[:, _phishing_idx(model)]
    raw_scores = (y_prob * 100).astype(int)
    boosted = np.array(
        [
            _apply_risk_floor(s, struct[i], df["text_clean"].iloc[i])
            for i, s in enumerate(raw_scores)
        ]
    )
    df["risk_score"] = boosted
    df["risk_level"] = df["risk_score"].apply(_map_risk_level)

    low_phish = df[(df["label"] == "phishing") & (df["risk_level"] == "LOW")]
    print(f"\n[전체 데이터셋 감사] LOW로 떨어진 phishing: {len(low_phish)}건")
    print("\n[source_dataset별]")
    print(low_phish["source_dataset"].value_counts())
    print("\n[call_type별]")
    print(low_phish["call_type"].value_counts())
    print("\n[샘플 5건]")
    print(
        low_phish[["source_dataset", "call_type", "risk_score", "text_clean"]]
        .sample(min(5, len(low_phish)), random_state=1)
        .to_string()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 저장 / 로드
# ─────────────────────────────────────────────────────────────────────────────


def save_artifacts(
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    vectorizer_type: str,
) -> None:
    """
    모델·임계값·클래스·벡터화 방식을 단일 아티팩트로 저장.
    FastAPI 서버 시작 시 load_artifacts() 한 번만 호출.

    SMS 모델(phishing_model_artifact.pkl)과 파일명 분리하여 혼용 방지.
    """
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "classes": list(model.classes_),
            "vectorizer_type": vectorizer_type,  # 추론 시 참조용
        },
        MODEL_PATH,
    )
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n[Save] {MODEL_PATH}, {VECTORIZER_PATH}")


def load_artifacts() -> tuple:
    """
    저장된 아티팩트 로드.
    FastAPI 서버 시작 시 1회 호출.

    Returns:
        (model, vectorizer, threshold, classes, vectorizer_type)
    """
    artifact = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return (
        artifact["model"],
        vectorizer,
        artifact["threshold"],
        artifact["classes"],
        artifact["vectorizer_type"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 추론 (FastAPI 연동용)
# ─────────────────────────────────────────────────────────────────────────────


def predict_risk_score(
    text_clean: str,
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    classes: list,
) -> dict:
    """
    전처리 완료된 STT 텍스트 → risk_score(0~100), risk_level 반환.
    구조적 위험 신호 하이브리드 보정(_apply_risk_floor) 적용 — NB가 신규
    어휘 때문에 낮게 잡아도 위험 신호가 뚜렷하면 최소 MEDIUM 이상 보장.
    """
    struct = _extract_struct_features(pd.Series([text_clean]))

    X = build_feature_matrix(vectorizer, pd.Series([text_clean]), struct, fit=False)
    prob_phish = model.predict_proba(X)[0][classes.index("phishing")]
    risk_score = int(prob_phish * 100)
    risk_score = _apply_risk_floor(risk_score, struct[0], text_clean)

    return {
        "risk_score": risk_score,
        "risk_level": _map_risk_level(risk_score),
        "modality": "voice",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df_train, df_val, df_test, df_holdout = load_data(DATA_PATH)

    best_vec_type, best_vectorizer, best_result = compare_vectorizers(df_train, df_val)
    if best_result["model"] is None:
        raise RuntimeError("유효한 모델 조합을 찾지 못했습니다.")

    struct_test = _extract_struct_features(df_test["text_clean"])
    X_test = build_feature_matrix(
        best_vectorizer, df_test["text_clean"], struct_test, fit=False
    )
    evaluate(best_result["model"], best_result["threshold"], X_test, df_test["label"])

    # 완전 신규 시나리오 일반화 + 오탐 검증
    evaluate_new_holdout(
        best_result["model"], best_vectorizer, best_result["threshold"], df_holdout
    )

    save_artifacts(
        best_result["model"], best_vectorizer, best_result["threshold"], best_vec_type
    )

    # 전체 데이터셋 사후 감사 — known 유형에서 LOW로 새는 phishing이 있는지 확인
    diagnose_low_phishing_full_dataset(
        best_result["model"], best_vectorizer, best_result["threshold"], DATA_PATH
    )


if __name__ == "__main__":
    main()
