"""동일 Holdout 데이터셋 대상 Stacking v1 vs v2 모델 성능 비교 및 McNemar 검정 스크립트"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from app.analysis.text.preprocessing import URL_PATTERN, normalize_text
from data_science.SMSModel.evaluation.metrics import (
    calculate_classification_metrics,
)
from data_science.SMSModel.template_grouping import create_text_fingerprint
from data_science.SMSModel.train_sms import DATA_PATH, load_data


SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_V1_MODEL_PATH = (
    SMS_MODEL_DIRECTORY / "artifacts" / "stacking" / "model.joblib"
)
DEFAULT_V2_MODEL_PATH = (
    SMS_MODEL_DIRECTORY / "artifacts" / "stacking" / "v2" / "model.joblib"
)
DEFAULT_OUTPUT_DIR = (
    SMS_MODEL_DIRECTORY / "reports" / "stacking_v2"
)
EXPECTED_HOLDOUT_COUNT = 210


def classify_pair(*, v1_correct: bool, v2_correct: bool) -> str:
    """v1과 v2의 정답 여부 쌍을 분류"""
    if v1_correct and v2_correct:
        return "both_correct"
    if not v1_correct and not v2_correct:
        return "both_wrong"
    if v1_correct:
        return "v1_only_correct"
    return "v2_only_correct"


def calculate_mcnemar_p_value(
    v1_only_correct: int, v2_only_correct: int
) -> float:
    """Exact McNemar test (Binomial Test) p-value 계산"""
    discordant = v1_only_correct + v2_only_correct
    if discordant == 0:
        return 1.0

    result = binomtest(
        min(v1_only_correct, v2_only_correct),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )
    return float(result.pvalue)


def load_model(model_path: Path) -> Any:
    """로컬 artifact의 hash를 확인하고 분류기 객체를 로드합니다."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    metadata_path = model_path.with_name("metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_sha256 = metadata.get("model_sha256")
        if expected_sha256 and expected_sha256 != calculate_sha256(model_path):
            raise ValueError(
                f"model checksum does not match metadata: {model_path}"
            )
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "classifier" in payload:
        return payload["classifier"]
    return payload


def calculate_sha256(path: Path) -> str:
    """파일의 SHA-256 hash를 반환합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_artifact_path(path: Path) -> str:
    """보고서에 로컬 절대 경로가 남지 않도록 표시 경로를 정규화합니다."""
    try:
        return path.resolve().relative_to(
            SMS_MODEL_DIRECTORY.resolve()
        ).as_posix()
    except ValueError:
        return path.name


def predict_model(
    classifier: Any,
    holdout_df: pd.DataFrame | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Artifact에 고정된 threshold로 확률과 이진 예측을 반환합니다."""
    if isinstance(holdout_df, pd.Series):
        holdout_df = pd.DataFrame({"text": holdout_df})

    if hasattr(classifier, "predict_probabilities"):
        probabilities, unavailable = classifier.predict_probabilities(
            holdout_df
        )
        if unavailable:
            raise RuntimeError(
                "base models unavailable during holdout comparison: "
                f"{unavailable}"
            )
        threshold = float(classifier.threshold)
        predictions = np.where(
            np.asarray(probabilities) >= threshold,
            "phishing",
            "normal",
        )
        return np.asarray(probabilities), predictions

    texts = holdout_df["text"]
    if hasattr(classifier, "predict_proba"):
        proba_matrix = classifier.predict_proba(texts)
        if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
            probabilities = proba_matrix[:, 1]
        else:
            probabilities = proba_matrix.ravel()
    elif hasattr(classifier, "predict_probability"):
        probabilities = classifier.predict_probability(texts)
    else:
        raise TypeError(
            "classifier does not expose a supported probability API"
        )

    predictions = classifier.predict(texts)
    return np.asarray(probabilities), np.asarray(predictions)


def load_holdout(holdout_path: Path | None = None) -> pd.DataFrame:
    """운영 데이터 CSV에서 학습에 제외된 공통 holdout을 복원합니다."""
    training_pool: pd.DataFrame | None = None
    if holdout_path is None:
        training_pool, holdout_df = load_data(DATA_PATH)
    else:
        holdout_df = pd.read_csv(holdout_path)

    required_columns = {"text", "label", "type"}
    missing = required_columns - set(holdout_df.columns)
    if missing:
        raise ValueError(f"holdout is missing columns: {sorted(missing)}")
    if holdout_path is None and len(holdout_df) != EXPECTED_HOLDOUT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_HOLDOUT_COUNT} holdout rows, "
            f"got {len(holdout_df)}"
        )

    result = holdout_df.copy().reset_index(drop=True)
    if "text_norm" not in result.columns:
        result["text_norm"] = result["text"].map(normalize_text)
    if "has_url" not in result.columns:
        result["has_url"] = result["text"].map(
            lambda text: bool(URL_PATTERN.search(str(text)))
        )
    if "text_fingerprint" not in result.columns:
        result["text_fingerprint"] = result["text_norm"].map(
            create_text_fingerprint
        )

    source_row_count = len(result)
    excluded_overlap_row_count = 0
    excluded_overlap_fingerprint_count = 0
    if training_pool is not None:
        overlap = set(training_pool["text_fingerprint"].astype(str)) & set(
            result["text_fingerprint"].astype(str)
        )
        if overlap:
            overlap_mask = result["text_fingerprint"].astype(str).isin(
                overlap
            )
            excluded_overlap_row_count = int(overlap_mask.sum())
            excluded_overlap_fingerprint_count = len(overlap)
            result = result.loc[~overlap_mask].reset_index(drop=True)

    if result.empty:
        raise ValueError("no leak-free holdout rows remain after validation")

    # DataFrame attrs는 원문이나 개인정보 없이 비교 보고서의 데이터 감사 내역으로 전달됩니다.
    result.attrs["holdout_audit"] = {
        "source_row_count": source_row_count,
        "evaluated_row_count": len(result),
        "excluded_training_overlap_rows": excluded_overlap_row_count,
        "excluded_training_overlap_fingerprints": (
            excluded_overlap_fingerprint_count
        ),
    }
    return result


def run_comparison(
    v1_model_path: Path,
    v2_model_path: Path,
    holdout_path: Path | None,
    output_dir: Path,
) -> Dict[str, Any]:
    """Holdout 데이터셋에 대해 v1과 v2를 비교하고 결과 CSV 및 요약 리포트 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 및 모델 로드
    holdout_df = load_holdout(holdout_path)
    holdout_audit = holdout_df.attrs.get(
        "holdout_audit",
        {
            "source_row_count": len(holdout_df),
            "evaluated_row_count": len(holdout_df),
            "excluded_training_overlap_rows": 0,
            "excluded_training_overlap_fingerprints": 0,
        },
    )
    v1_model = load_model(v1_model_path)
    v2_model = load_model(v2_model_path)

    labels = holdout_df["label"]

    # 두 모델은 artifact에 저장된 각자의 고정 threshold를 그대로 사용합니다.
    v1_prob, v1_pred = predict_model(v1_model, holdout_df)
    v2_prob, v2_pred = predict_model(v2_model, holdout_df)

    v1_correct = v1_pred == labels.values
    v2_correct = v2_pred == labels.values

    # 표본 단위 비교 분류 생성
    comparison_series = [
        classify_pair(v1_correct=c1, v2_correct=c2)
        for c1, c2 in zip(v1_correct, v2_correct)
    ]

    # 결과 DataFrame 생성
    result_df = pd.DataFrame(
        {
            "text_fingerprint": holdout_df.get(
                "text_fingerprint", holdout_df.index.astype(str)
            ),
            "label": labels,
            "type": holdout_df.get("type", "unknown"),
            "v1_probability": v1_prob,
            "v1_prediction": v1_pred,
            "v1_correct": v1_correct,
            "v2_probability": v2_prob,
            "v2_prediction": v2_pred,
            "v2_correct": v2_correct,
            "comparison": comparison_series,
        }
    )

    # CSV 저장
    csv_output_path = output_dir / "holdout_comparison_predictions.csv"
    result_df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    # McNemar Test 및 통계 요약
    both_correct = int((result_df["comparison"] == "both_correct").sum())
    both_wrong = int((result_df["comparison"] == "both_wrong").sum())
    v1_only_correct = int((result_df["comparison"] == "v1_only_correct").sum())
    v2_only_correct = int((result_df["comparison"] == "v2_only_correct").sum())

    p_value = calculate_mcnemar_p_value(v1_only_correct, v2_only_correct)

    v1_metrics = calculate_classification_metrics(labels, v1_pred)
    v2_metrics = calculate_classification_metrics(labels, v2_pred)

    summary = {
        "sample_count": len(result_df),
        "holdout_audit": holdout_audit,
        "evaluation_policy": {
            "dataset": "shared_holdout",
            "used_for_training": False,
            "used_for_threshold_selection": False,
            "thresholds_changed_after_evaluation": False,
        },
        "artifacts": {
            "v1": {
                "path": portable_artifact_path(v1_model_path),
                "sha256": calculate_sha256(v1_model_path),
                "threshold": getattr(v1_model, "threshold", None),
            },
            "v2": {
                "path": portable_artifact_path(v2_model_path),
                "sha256": calculate_sha256(v2_model_path),
                "threshold": getattr(v2_model, "threshold", None),
            },
        },
        "v1_metrics": v1_metrics.to_dict(),
        "v2_metrics": v2_metrics.to_dict(),
        # 기존 소비자와 이전 테스트를 위한 명시적 요약 필드입니다.
        "v1_accuracy": v1_metrics.accuracy,
        "v2_accuracy": v2_metrics.accuracy,
        "contingency_table": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "v1_only_correct": v1_only_correct,
            "v2_only_correct": v2_only_correct,
        },
        "mcnemar_test": {
            "discordant_count": v1_only_correct + v2_only_correct,
            "p_value": p_value,
            "is_significant_at_0_05": p_value < 0.05,
        },
    }

    # 요약 JSON 저장
    json_output_path = output_dir / "holdout_comparison_summary.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stacking Model v1 vs v2 Holdout Comparison"
    )
    parser.add_argument(
        "--v1-model-path", type=Path, default=DEFAULT_V1_MODEL_PATH
    )
    parser.add_argument(
        "--v2-model-path", type=Path, default=DEFAULT_V2_MODEL_PATH
    )
    parser.add_argument(
        "--holdout-path",
        type=Path,
        default=None,
        help="Optional test fixture CSV; defaults to the 210-row holdout in DATA_PATH.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )

    args = parser.parse_args()
    summary = run_comparison(
        v1_model_path=args.v1_model_path,
        v2_model_path=args.v2_model_path,
        holdout_path=args.holdout_path,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
