"""동일 Holdout 데이터셋 대상 Stacking v1 vs v2 모델 성능 비교 및 McNemar 검정 스크립트"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# 기본 경로 정의 
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_V1_MODEL_PATH = BASE_DIR / "artifacts" / "stacking" / "model.joblib"
DEFAULT_V2_MODEL_PATH = BASE_DIR / "artifacts" / "stacking" / "v2" / "model.joblib"
DEFAULT_HOLDOUT_PATH = BASE_DIR / "data" / "holdout_210.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "artifacts" / "stacking" / "comparison"


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
    """joblib payload에서 분류기 객체 로드"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "classifier" in payload:
        return payload["classifier"]
    return payload


def predict_model(classifier: Any, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """모델의 예측 확률(피싱 확률)과 예측 레이블 반환"""
    # predict_proba 지원 여부 확인
    if hasattr(classifier, "predict_proba"):
        proba_matrix = classifier.predict_proba(texts)
        if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
            probabilities = proba_matrix[:, 1]
        else:
            probabilities = proba_matrix.ravel()
    elif hasattr(classifier, "predict_probability"):
        probabilities = classifier.predict_probability(texts)
    else:
        probabilities = np.full(len(texts), np.nan)

    predictions = classifier.predict(texts)
    return np.asarray(probabilities), np.asarray(predictions)


def run_comparison(
    v1_model_path: Path,
    v2_model_path: Path,
    holdout_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Holdout 데이터셋에 대해 v1과 v2를 비교하고 결과 CSV 및 요약 리포트 생성"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 및 모델 로드
    holdout_df = pd.read_csv(holdout_path)
    v1_model = load_model(v1_model_path)
    v2_model = load_model(v2_model_path)

    texts = holdout_df["text"]
    labels = holdout_df["label"]

    # 모델 예측 수행 
    v1_prob, v1_pred = predict_model(v1_model, texts)
    v2_prob, v2_pred = predict_model(v2_model, texts)

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

    v1_acc = float(v1_correct.mean())
    v2_acc = float(v2_correct.mean())

    summary = {
        "sample_count": len(result_df),
        "v1_accuracy": v1_acc,
        "v2_accuracy": v2_acc,
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
        "--holdout-path", type=Path, default=DEFAULT_HOLDOUT_PATH
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