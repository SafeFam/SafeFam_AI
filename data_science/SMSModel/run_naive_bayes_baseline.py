"""Naive Bayes text-only 및 structural baseline 실행기"""

from __future__ import annotations

from pathlib import Path

from data_science.SMSModel.evaluation import (
    ModelEvaluationResult,
    save_model_evaluation_reports,
    train_and_evaluate_model,
)
from data_science.SMSModel.modeling import (
    NaiveBayesPhishingClassifier,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    load_data,
    split_data,
)

SMS_MODEL_DIR = Path(__file__).resolve().parent

NB_REPORT_DIRECTORY = (
    SMS_MODEL_DIR / "reports" / "model_evaluation" / "naive_bayes_baseline"
)

TARGET_PHISHING_RECALL = 0.96
LATENCY_SAMPLE_COUNT = 100


def evaluate_naive_bayes_baselines() -> list[ModelEvaluationResult]:
    """동일한 leakage-safe split에서 두 NB 모델을 평가"""

    dataset, _holdout = load_data(DATA_PATH)

    splits = split_data(dataset)

    text_only_model = NaiveBayesPhishingClassifier(
        include_structural_features=False,
    )

    structural_model = NaiveBayesPhishingClassifier(
        include_structural_features=True,
    )

    text_only_result = train_and_evaluate_model(
        text_only_model,
        train_df=splits.train,
        validation_df=splits.validation,
        test_df=splits.test,
        target_recall=(TARGET_PHISHING_RECALL),
        latency_sample_count=(LATENCY_SAMPLE_COUNT),
    )

    structural_result = train_and_evaluate_model(
        structural_model,
        train_df=splits.train,
        validation_df=splits.validation,
        test_df=splits.test,
        target_recall=(TARGET_PHISHING_RECALL),
        latency_sample_count=(LATENCY_SAMPLE_COUNT),
    )

    results = [
        text_only_result,
        structural_result,
    ]

    save_model_evaluation_reports(
        results,
        output_directory=NB_REPORT_DIRECTORY,
    )

    # 공식 운영 baseline은 structural 모델
    return results


def print_results(
    results: list[ModelEvaluationResult],
) -> None:
    """터미널에서 핵심 평가 결과를 출력"""

    print("\n[Naive Bayes Baseline]")

    for result in results:
        metrics = result.test_metrics

        baseline_marker = (
            " [official baseline]"
            if result.model_name == "naive_bayes_structural"
            else " [ablation]"
        )

        print(
            f"- {result.model_name}"
            f"{baseline_marker}\n"
            f"  threshold="
            f"{result.selected_threshold:.6f} | "
            f"precision={metrics.precision:.4f} | "
            f"recall={metrics.recall:.4f} | "
            f"f1={metrics.f1:.4f} | "
            f"f2={metrics.f2:.4f} | "
            f"FN={metrics.false_negative} | "
            f"avg_ms="
            f"{result.latency.average_ms:.3f}"
        )


def main() -> None:
    results = evaluate_naive_bayes_baselines()
    print_results(results)

    print(f"\n[Report] {NB_REPORT_DIRECTORY}")
    print(f"[Artifact] unchanged: {MODEL_PATH}")
    print(f"[Vectorizer] unchanged: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
