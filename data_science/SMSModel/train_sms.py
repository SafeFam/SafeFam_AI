"""SMS 피싱 탐지 베이즈 분류기 학습 파이프라인"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB

from app.analysis.text.preprocessing import (
    extract_struct_feature_matrix,
    normalize_text,
)
from data_science.SMSModel.dataset_splitting import (
    DatasetSplitConfig,
    DatasetSplits,
    load_split_manifest,
    save_split_manifest,
    split_grouped_dataset,
    validate_dataset_splits,
)
from data_science.SMSModel.reporting import (
    generate_dataset_split_reports,
)
from data_science.SMSModel.template_grouping import (
    TemplateGroupingConfig,
    add_text_fingerprints,
    prepare_template_groups,
)
from data_science.SMSModel.data_quality import (
    validate_sms_dataset,
)

warnings.filterwarnings("ignore")

ALLOWED_DATA_SOURCES = {
    "original",
    "user_added",
    "synthetic_new_holdout",
    "synthetic_fp_stress",
    "synthetic_fp_stress_train",
    "reviewed_reclassification_v2",
    "public_phishing_v2",
    "synthetic_diversity_v2",
    "synthetic_hard_negative_v2",
    "synthetic_normal_v3",
    "real_collected_v4",
    "real_holdout",
}

REAL_HOLDOUT_SOURCES = ("real_holdout",)
SYNTHETIC_STRESS_SOURCES = (
    "synthetic_new_holdout",
    "synthetic_fp_stress",
)
HOLDOUT_SOURCES = REAL_HOLDOUT_SOURCES + SYNTHETIC_STRESS_SOURCES


# ─────────────────────────────────────────────────────────────────────────────
# 경로 CONFIG — 실행 위치가 아니라 이 파일의 위치를 기준으로 계산
# ─────────────────────────────────────────────────────────────────────────────

SMS_MODEL_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    SMS_MODEL_DIR.parent / "Data" / "SMSData" / "phishing_total_dataset_reclassified.csv"
)
ARTIFACTS_DIR = SMS_MODEL_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "phishing_model_artifact.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "phishing_vectorizer.pkl"
SPLIT_MANIFEST_PATH = SMS_MODEL_DIR / "splits" / "sms_split_v5.csv"

# 보고서 경로
REPORTS_DIR = SMS_MODEL_DIR / "reports"

DATASET_SPLIT_JSON_REPORT_PATH = REPORTS_DIR / "dataset_split_summary.json"

DATASET_SPLIT_MARKDOWN_REPORT_PATH = REPORTS_DIR / "dataset_split_summary.md"

# ─────────────────────────────────────────────────────────────────────────────
# 학습 CONFIG
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
VAL_SIZE = 0.15  
TEST_SIZE = 0.15 
TARGET_PHISHING_RECALL = 0.96
FALLBACK_RECALL_TOLERANCE = 0.02

RISK_HIGH_THRESHOLD = 70 
RISK_MEDIUM_THRESHOLD = 40 

# 격자 탐색 범위
ALPHA_GRID = [1.5, 2.0, 3.0, 5.0]
THRESHOLD_GRID = np.round(np.arange(0.30, 0.75, 0.05), 2)

# ─────────────────────────────────────────────────────────────────────────────
# 유사 템플릿 그룹화 CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# 유사도 임계값
TEMPLATE_SIMILARITY_THRESHOLD = 0.88

# n-gram 범위
TEMPLATE_NGRAM_RANGE = (2, 5)

# TF-IDF 최대 피처 수
TEMPLATE_MAX_FEATURES = 50_000


def build_template_grouping_config() -> TemplateGroupingConfig:
    """현재 학습 실행에서 사용할 템플릿 그룹화 설정을 반환"""
    return TemplateGroupingConfig(
        similarity_threshold=TEMPLATE_SIMILARITY_THRESHOLD,
        ngram_range=TEMPLATE_NGRAM_RANGE,
        min_df=1,
        max_features=TEMPLATE_MAX_FEATURES,
    )


def build_dataset_split_config() -> DatasetSplitConfig:
    """현재 SMS 모델 학습에서 사용할 데이터 분할 설정"""
    return DatasetSplitConfig(
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        candidate_count=500,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────


def select_real_holdout(holdout: pd.DataFrame) -> pd.DataFrame:
    """실제 문자로 구성된 주 평가셋만 반환"""
    if "source" not in holdout.columns:
        return holdout.iloc[0:0]
    return holdout[
        holdout["source"].isin(REAL_HOLDOUT_SOURCES)
    ].reset_index(drop=True)


def select_synthetic_stress(holdout: pd.DataFrame) -> pd.DataFrame:
    """합성 FP 스트레스 셋만 반환한다. 보조 지표로만 사용"""
    if "source" not in holdout.columns:
        return holdout.iloc[0:0]
    return holdout[
        holdout["source"].isin(SYNTHETIC_STRESS_SOURCES)
    ].reset_index(drop=True)


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSV를 읽고 학습용 데이터와 별도 holdout 데이터를 반환"""
    df = pd.read_csv(path)

    df = validate_sms_dataset(
        df,
        allowed_sources=ALLOWED_DATA_SOURCES,
    )

    required_columns = {
        "text",
        "label",
        "type",
        "has_url",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"누락 컬럼: {missing_columns}")

    if df.isnull().any().any():
        raise ValueError("결측치가 존재합니다.")

    allowed_labels = {"phishing", "normal"}
    actual_labels = set(df["label"].unique())

    if not allowed_labels.issuperset(actual_labels):
        raise ValueError(f"예상치 못한 label 값: {df['label'].unique()}")

    print(f"[Load] 원본 {len(df)}건 | {df['label'].value_counts().to_dict()}")

    # 학습과 API가 공유하는 공통 정규화 함수 사용
    df["text_norm"] = df["text"].apply(normalize_text)

    source = df.get(
        "source",
        pd.Series("original", index=df.index),
    )

    is_new_holdout = source.isin(HOLDOUT_SOURCES)

    # 신규 시나리오 holdout은 학습 데이터 그룹화 대상에서도 제외
    df_holdout = df[is_new_holdout].copy().reset_index(drop=True)

    # 평가셋도 완전 중복 제거
    df_holdout = add_text_fingerprints(df_holdout, text_column="text_norm")
    holdout_before_deduplication = len(df_holdout)
    df_holdout = df_holdout.drop_duplicates(
        subset="text_fingerprint",
        keep="first",
    ).reset_index(drop=True)

    df_pool = df[~is_new_holdout].copy().reset_index(drop=True)

    before_deduplication = len(df_pool)

    # fingerprint 생성 → label 충돌 검사 → 완전 중복 제거 → 유사 그룹화
    df_pool = prepare_template_groups(
        df_pool,
        config=build_template_grouping_config(),
        text_column="text_norm",
        label_column="label",
    )

    removed_duplicates = before_deduplication - len(df_pool)
    template_group_count = df_pool["template_group_id"].nunique()

    group_sizes = df_pool["template_group_id"].value_counts()
    similar_group_count = int((group_sizes > 1).sum())
    largest_group_size = int(group_sizes.max()) if not group_sizes.empty else 0

    print(
        f"[Dedup] fingerprint 기준 완전 중복 제거: "
        f"{before_deduplication} → {len(df_pool)}건 "
        f"(제거 {removed_duplicates}건)"
    )

    print(
        f"[Template Grouping] 전체 그룹={template_group_count} | "
        f"유사 메시지 그룹={similar_group_count} | "
        f"최대 그룹 크기={largest_group_size}"
    )

    print(f"[Pool] label 분포: {df_pool['label'].value_counts().to_dict()}")

    print(
        f"[Holdout] 평가셋 {holdout_before_deduplication} → "
        f"{len(df_holdout)}건 분리 (학습에 전혀 사용 안 됨)"
    )
    print(
        "[Holdout] source 분포: "
        f"{df_holdout['source'].value_counts().to_dict()}"
    )

    return df_pool, df_holdout


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test 분리
# ─────────────────────────────────────────────────────────────────────────────


def split_data(
    df: pd.DataFrame,
    *,
    create_manifest: bool = False,
) -> DatasetSplits:
    """저장된 manifest를 사용하거나 새로운 그룹 split을 생성"""
    split_config = build_dataset_split_config()
    grouping_config = build_template_grouping_config()

    if SPLIT_MANIFEST_PATH.exists() and not create_manifest:
        splits = load_split_manifest(
            df,
            SPLIT_MANIFEST_PATH,
            config=split_config,
        )

        print(f"[Split] 기존 manifest 사용: {SPLIT_MANIFEST_PATH}")
    else:
        splits = split_grouped_dataset(
            df,
            config=split_config,
        )

        validate_dataset_splits(
            df,
            splits,
            config=split_config,
        )

        save_split_manifest(
            splits,
            SPLIT_MANIFEST_PATH,
            overwrite=create_manifest,
        )

        print(f"[Split] 새 manifest 저장: {SPLIT_MANIFEST_PATH}")

    # manifest를 로드한 경우에도 학습 직전에 다시 검증
    validate_dataset_splits(df, splits, config=split_config)

    summary = generate_dataset_split_reports(
        df,
        splits,
        split_config=split_config,
        grouping_config=grouping_config,
        json_path=DATASET_SPLIT_JSON_REPORT_PATH,
        markdown_path=DATASET_SPLIT_MARKDOWN_REPORT_PATH,
    )

    def describe_split(
        name: str,
        split_df: pd.DataFrame,
    ) -> None:
        label_counts = split_df["label"].value_counts()
        group_count = split_df["template_group_id"].nunique()

        print(
            f"[Split] {name:<10} "
            f"rows={len(split_df)} | "
            f"groups={group_count} | "
            f"phishing={label_counts.get('phishing', 0)} "
            f"({(split_df['label'] == 'phishing').mean():.1%}) | "
            f"normal={label_counts.get('normal', 0)} "
            f"({(split_df['label'] == 'normal').mean():.1%})"
        )

    describe_split("Train", splits.train)
    describe_split("Validation", splits.validation)
    describe_split("Test", splits.test)

    print(
        "[Split Validation] "
        f"passed={summary['validation']['passed']} | "
        f"group_overlap={summary['validation']['group_overlap_count']} | "
        "fingerprint_overlap="
        f"{summary['validation']['fingerprint_overlap_count']}"
    )
    print(f"[Dataset] fingerprint={summary['dataset_fingerprint']}")
    print(f"[Report] {DATASET_SPLIT_JSON_REPORT_PATH}")
    print(f"[Report] {DATASET_SPLIT_MARKDOWN_REPORT_PATH}")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 벡터화
# ─────────────────────────────────────────────────────────────────────────────


def build_vectorizer() -> CountVectorizer:
    """char_wb(단어 경계 문자 n-gram): 형태소 분석기 없이 한글 조사 변형 대응"""
    return CountVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.95,
        max_features=8_000,
    )


def build_feature_matrix(
    vectorizer: CountVectorizer,
    text_norm: pd.Series,
    struct: np.ndarray,
    *,
    fit: bool,
):
    """
    텍스트 피처(CountVectorizer) + 구조적 피처(6개) sparse hstack 결합"""
    X_text = (
        vectorizer.fit_transform(text_norm) if fit else vectorizer.transform(text_norm)
    )
    X_struct = csr_matrix(struct)
    return hstack([X_text, X_struct])


# ─────────────────────────────────────────────────────────────────────────────
# 학습 및 튜닝
# ─────────────────────────────────────────────────────────────────────────────


def _phishing_idx(model) -> int:
    """model.classes_ 에서 'phishing' 인덱스 반환."""
    return list(model.classes_).index("phishing")


def _compute_recalls(
    model, X_test, y_test: pd.Series, threshold: float
) -> tuple[float, float]:
    """threshold 적용 후 Recall(phishing), Recall(normal) 반환."""
    y_prob = model.predict_proba(X_test)[:, _phishing_idx(model)]
    y_pred = np.where(y_prob >= threshold, "phishing", "normal")
    cm = confusion_matrix(y_test, y_pred, labels=["normal", "phishing"])

    rec_p = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    rec_n = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    return rec_p, rec_n


def train_and_tune(
    X_train,
    y_train: pd.Series,
    X_val,
    y_val: pd.Series,
) -> dict:
    """
    [격자 탐색] alpha × threshold 전체 탐색.
    튜닝 기준: validation 세트 사용 (test는 최종 평가 전용으로 봉인 — test set 누수 방지)

    v2 변경: ComplementNB를 CalibratedClassifierCV(isotonic)로 감쌈.
    - isotonic: 비선형 보정 — 작은 데이터셋에서 sigmoid보다 안정적
    - cv=5: 5-fold 교차검증으로 보정 파라미터 추정
    - 보정 후 prob_phishing이 0~1 사이에 고르게 분산 → 상/중/하 구분 가능

    선택 기준:
      1순위) Recall(phishing) >= TARGET 조건 하 Recall(normal) 최대
      2순위) 목표 미달 시 Recall(phishing) 최대 (fallback)
    """
    best: dict = {
        "model": None,
        "alpha": None,
        "threshold": None,
        "recall_phishing": 0.0,
        "recall_normal": 0.0,
    }

    header = f"{'alpha':>6} | {'thresh':>6} | {'rec_phish':>10} | {'rec_normal':>10}"
    print(
        f"\n{'=' * 60}\n[ 격자 탐색: alpha × threshold (CalibratedComplementNB, val 기준) ]\n{'=' * 60}"
    )
    print(header)
    print("-" * len(header))

    for alpha in ALPHA_GRID:
        base_model = ComplementNB(alpha=alpha)
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic", 
            cv=5,  
        )
        calibrated.fit(X_train, y_train)

        for threshold in THRESHOLD_GRID:
            rec_p, rec_n = _compute_recalls(calibrated, X_val, y_val, threshold)

            print(f"{alpha:>6} | {threshold:>6.2f} | {rec_p:>10.4f} | {rec_n:>10.4f}")

            candidate_meets_target = rec_p >= TARGET_PHISHING_RECALL
            best_meets_target = (
                best["model"] is not None
                and best["recall_phishing"] >= TARGET_PHISHING_RECALL
            )
            should_replace = (
                best["model"] is None
                or (
                    candidate_meets_target
                    and (not best_meets_target or rec_n >= best["recall_normal"])
                )
                or (
                    not candidate_meets_target
                    and not best_meets_target
                    and (
                        rec_p > best["recall_phishing"] + FALLBACK_RECALL_TOLERANCE
                        # recall(phishing)이 (근사)동률이면 threshold를 낮춰 무작정 더
                        # 많이 잡으려 하기보다 recall(normal)이 더 나은(=오탐이 적은)
                        # 쪽을 선택한다 - target 미달 시 fallback이 낮은 threshold만
                        # 고수해 일상 대화까지 의심 문자로 분류하던 문제를 방지한다.
                        or (
                            abs(rec_p - best["recall_phishing"])
                            <= FALLBACK_RECALL_TOLERANCE
                            and rec_n > best["recall_normal"]
                        )
                    )
                )
            )

            if should_replace:
                best.update(
                    {
                        "model": calibrated,
                        "alpha": alpha,
                        "threshold": threshold,
                        "recall_phishing": rec_p,
                        "recall_normal": rec_n,
                    }
                )

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 보정 후 확률 분포 검증
# ─────────────────────────────────────────────────────────────────────────────


def verify_probability_distribution(model, X_test, y_test: pd.Series) -> None:
    """보정 후 prob_phishing 분포를 확인"""
    probs = model.predict_proba(X_test)[:, _phishing_idx(model)]
    df_prob = pd.DataFrame({"prob": probs, "label": y_test.values})

    low = (probs < 0.40).sum()
    medium = ((probs >= 0.40) & (probs < 0.70)).sum()
    high = (probs >= 0.70).sum()

    print(f"\n{'=' * 60}\n[ 보정 후 prob_phishing 분포 검증 ]\n{'=' * 60}")
    print(f"LOW    (<0.40) : {low:>4}건 ({low / len(probs):.1%})")
    print(f"MEDIUM (0.40~0.70): {medium:>4}건 ({medium / len(probs):.1%})")
    print(f"HIGH   (≥0.70) : {high:>4}건 ({high / len(probs):.1%})")
    print("\nlabel별 평균 prob_phishing:")
    print(df_prob.groupby("label")["prob"].describe().round(4))

    if medium == 0:
        print("[WARNING] MEDIUM 구간 샘플 없음 — 보정 방식 재검토 필요")


# ─────────────────────────────────────────────────────────────────────────────
# 최종 평가
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, threshold: float, X_test, y_test: pd.Series) -> None:
    """최적 threshold 적용 후 classification report + confusion matrix 출력"""
    y_prob = model.predict_proba(X_test)[:, _phishing_idx(model)]
    y_pred = np.where(y_prob >= threshold, "phishing", "normal")

    print(f"\n{'=' * 60}\n[ 최종 평가 — threshold={threshold} ]\n{'=' * 60}")
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
            f"[WARNING] Recall(phishing) {rec_p:.4f} < 목표 {TARGET_PHISHING_RECALL:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 저장 / 로드
# ─────────────────────────────────────────────────────────────────────────────


def save_artifacts(model, vectorizer: CountVectorizer, threshold: float) -> None:
    """model + threshold + classes를 단일 아티팩트로 저장 (FastAPI 로드용)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "classes": list(model.classes_)},
        MODEL_PATH,
    )
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n[Save] {MODEL_PATH}, {VECTORIZER_PATH}")


def load_artifacts() -> tuple:
    """
    FastAPI 서버 시작 시 1회 호출.
    Returns: (model, vectorizer, threshold, classes)
    """
    artifact = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return artifact["model"], vectorizer, artifact["threshold"], artifact["classes"]


# ─────────────────────────────────────────────────────────────────────────────
# 추론 (FastAPI 연동용)
# ─────────────────────────────────────────────────────────────────────────────


def _map_risk_level(risk_score: int) -> str:
    """risk_score → risk_level 매핑"""
    if risk_score >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    if risk_score >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def predict_risk_score(
    text: str,
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    classes: list,
) -> dict:
    """단일 SMS를 분석해 위험 점수와 위험 등급을 반환"""
    text_norm = normalize_text(text)

    struct = extract_struct_feature_matrix([text])

    X = build_feature_matrix(
        vectorizer,
        pd.Series([text_norm]),
        struct,
        fit=False,
    )

    phishing_index = classes.index("phishing")
    phishing_probability = model.predict_proba(X)[0][phishing_index]
    risk_score = int(phishing_probability * 100)

    return {
        "risk_score": risk_score,
        "risk_level": _map_risk_level(risk_score),
        "text_score_only": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    df, df_holdout = load_data(DATA_PATH)

    splits = split_data(df)
    df_train = splits.train
    df_val = splits.validation
    df_test = splits.test

    struct_train = extract_struct_feature_matrix(df_train["text"])
    struct_val = extract_struct_feature_matrix(df_val["text"])
    struct_test = extract_struct_feature_matrix(df_test["text"])

    vectorizer = build_vectorizer()
    X_train = build_feature_matrix(
        vectorizer, df_train["text_norm"], struct_train, fit=True
    )
    X_val = build_feature_matrix(vectorizer, df_val["text_norm"], struct_val, fit=False)
    X_test = build_feature_matrix(
        vectorizer, df_test["text_norm"], struct_test, fit=False
    )

    best = train_and_tune(X_train, df_train["label"], X_val, df_val["label"])
    if best["model"] is None:
        raise RuntimeError("유효한 모델 조합을 찾지 못했습니다.")

    print(
        f"\n★ 최적: CalibratedComplementNB | alpha={best['alpha']} "
        f"| threshold={best['threshold']}\n"
        f"  Recall(phishing)={best['recall_phishing']:.4f} | "
        f"Recall(normal)={best['recall_normal']:.4f}"
    )

    verify_probability_distribution(best["model"], X_test, df_test["label"])
    evaluate(best["model"], best["threshold"], X_test, df_test["label"])

    # 실제 문자 평가셋과 합성 스트레스 셋은 성격이 다르므로 따로 보고한다.
    evaluate_new_holdout(
        best["model"],
        vectorizer,
        best["threshold"],
        select_real_holdout(df_holdout),
        title="실제 문자 주 평가셋",
    )
    evaluate_new_holdout(
        best["model"],
        vectorizer,
        best["threshold"],
        select_synthetic_stress(df_holdout),
        title="합성 FP 스트레스 셋 (보조 지표)",
    )

    save_artifacts(best["model"], vectorizer, best["threshold"])


def evaluate_new_holdout(
    model,
    vectorizer: CountVectorizer,
    threshold: float,
    df_holdout: pd.DataFrame,
    *,
    title: str = "신규 시나리오",
) -> None:
    """
    학습에 전혀 관여하지 않은 평가셋으로 일반화 성능 + 오탐률을 검증.

    주 평가셋(real_holdout)과 보조 스트레스 셋(synthetic_*)은 분포가 달라
    지표를 합치면 해석이 흐려지므로 호출부에서 나눠 전달한다.
    """
    if df_holdout.empty:
        print(f"\n[SKIP] {title} 데이터 없음")
        return

    text_norm = df_holdout["text"].apply(normalize_text)
    struct = extract_struct_feature_matrix(df_holdout["text"])
    X = build_feature_matrix(vectorizer, text_norm, struct, fit=False)

    y_prob = model.predict_proba(X)[:, _phishing_idx(model)]
    y_pred = np.where(y_prob >= threshold, "phishing", "normal")

    print(
        f"\n{'=' * 60}\n[ holdout 평가 · {title} — {len(df_holdout)}건 ]\n{'=' * 60}"
    )
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
    df_h["risk_score"] = (y_prob * 100).astype(int)

    phishing_df = df_h[df_h["label"] == "phishing"]
    if not phishing_df.empty:
        print("\n[유형별 Recall(phishing)] — 낮은 순")
        print(
            phishing_df.groupby("type")
            .apply(lambda g: (g["pred"] == "phishing").mean())
            .sort_values()
        )

    normal_df = df_h[df_h["label"] == "normal"]
    if not normal_df.empty:
        print("\n[유형별 오탐률(FP)]")
        print(
            normal_df.groupby("type")
            .apply(lambda g: (g["pred"] == "phishing").mean())
            .sort_values(ascending=False)
        )


if __name__ == "__main__":
    main()
