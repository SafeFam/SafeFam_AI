"""모델 평가 결과 JSON, CSV, Markdown 출력"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .evaluator import ModelEvaluationResult


def save_model_evaluation_json(
    results: list[ModelEvaluationResult],
    path: Path,
) -> None:
    """상세 평가 결과를 JSON으로 저장"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "models": [result.to_dict() for result in results],
    }

    # 한글 깨짐 방지
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_model_evaluation_csv(
    results: list[ModelEvaluationResult],
    path: Path,
) -> None:
    """모델별 핵심 성능 비교표를 CSV로 저장"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV 컬럼 레이아웃 정의
    fieldnames = [
        "model_name",
        "score_type",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
        "average_inference_ms",
        "p95_inference_ms",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        # 각 모델의 핵심 지표만 추출하여 행 단위로 기록
        for result in results:
            metrics = result.test_metrics

            writer.writerow(
                {
                    "model_name": result.model_name,
                    "score_type": result.score_type,
                    "threshold": (result.selected_threshold),
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "f2": metrics.f2,
                    "false_positive": (metrics.false_positive),
                    "false_negative": (metrics.false_negative),
                    "true_positive": (metrics.true_positive),
                    "true_negative": (metrics.true_negative),
                    "average_inference_ms": (result.latency.average_ms),
                    "p95_inference_ms": (result.latency.p95_ms),
                }
            )


def render_model_evaluation_markdown(
    results: list[ModelEvaluationResult],
) -> str:
    """모델 비교 결과를 Markdown 표로 렌더링"""
    lines = [
        "# Phishing Model Evaluation",
        "",
        "| Model | Threshold | Precision | Recall | F1 | F2 | FN | FP | Avg ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        metrics = result.test_metrics
        latency = result.latency

        # 소수점 자릿수 정렬
        lines.append(
            f"| {result.model_name} "
            f"| {result.selected_threshold:.6f} "
            f"| {metrics.precision:.4f} "
            f"| {metrics.recall:.4f} "
            f"| {metrics.f1:.4f} "
            f"| {metrics.f2:.4f} "
            f"| {metrics.false_negative} "
            f"| {metrics.false_positive} "
            f"| {latency.average_ms:.3f} "
            f"| {latency.p95_ms:.3f} |"
        )

    return "\n".join(lines) + "\n"


def save_model_evaluation_markdown(
    results: list[ModelEvaluationResult],
    path: Path,
) -> None:
    """Markdown 모델 비교표를 저장"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        render_model_evaluation_markdown(results),
        encoding="utf-8",
    )


def save_model_evaluation_reports(
    results: list[ModelEvaluationResult],
    *,
    output_directory: Path,
) -> None:
    """JSON, CSV, Markdown 결과를 모두 저장"""
    if not results:
        raise ValueError("at least one evaluation result is required")

    save_model_evaluation_json(
        results,
        output_directory / "model_evaluation.json",
    )
    save_model_evaluation_csv(
        results,
        output_directory / "model_evaluation.csv",
    )
    save_model_evaluation_markdown(
        results,
        output_directory / "model_evaluation.md",
    )
