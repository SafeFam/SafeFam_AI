"""Naive Bayes, Logistic Regression, Linear SVM 통합 비교 실행기"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from data_science.SMSModel.evaluation import (
    ModelEvaluationResult,
    save_model_evaluation_reports,
    train_and_evaluate_model,
)
from data_science.SMSModel.evaluation.reporting import (
    render_model_evaluation_markdown,
)
from data_science.SMSModel.modeling import (
    BasePhishingClassifier,
    LinearSvmPhishingClassifier,
    LogisticRegressionPhishingClassifier,
    NaiveBayesPhishingClassifier,
    save_comparison_artifact,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    DATASET_SPLIT_JSON_REPORT_PATH,
    SPLIT_MANIFEST_PATH,
    TARGET_PHISHING_RECALL,
    load_data,
    split_data,
)

SMS_MODEL_DIR = Path(__file__).resolve().parent

# 통합 비교 보고서 저장 위치
COMPARISON_REPORT_DIRECTORY = (
    SMS_MODEL_DIR
    / "reports"
    / "model_evaluation"
    / "model_comparison"
)

# 신규 모델 비교 artifact 저장 위치
COMPARISON_ARTIFACT_DIRECTORY = (
    SMS_MODEL_DIR
    / "artifacts"
    / "comparison"
)

LOGISTIC_ARTIFACT_DIRECTORY = (
    COMPARISON_ARTIFACT_DIRECTORY
    / "logistic_regression"
)

LINEAR_SVM_ARTIFACT_DIRECTORY = (
    COMPARISON_ARTIFACT_DIRECTORY
    / "linear_svm"
)

# 공통 평가기의 단건 추론시간 측정 개수
LATENCY_SAMPLE_COUNT = 100

# 실행 단위 보고서의 schema version
COMPARISON_RUN_SCHEMA_VERSION = 1

# 현재 committed split manifest의 파일명 기반 버전
EXPECTED_SPLIT_MANIFEST_VERSION = "sms_split_v2"


def _get_package_version(package_name: str) -> str:
    """보고서에 기록할 설치 패키지 버전을 반환합니다."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _collect_execution_environment() -> dict[str, Any]:
    """latency 해석에 필요한 실행환경을 개인정보 없이 수집합니다."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "library_versions": {
            "numpy": _get_package_version("numpy"),
            "pandas": _get_package_version("pandas"),
            "scikit_learn": _get_package_version("scikit-learn"),
            "kiwipiepy": _get_package_version("kiwipiepy"),
        },
    }


def _calculate_sha256(path: Path) -> str:
    """파일 내용을 기준으로 SHA-256 fingerprint를 계산"""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(
            lambda: source_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _require_committed_split_manifest() -> None:
    """고정된 split manifest가 없으면 임의 재분할을 막기 위해 중단"""
    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "committed split manifest is required: "
            f"{SPLIT_MANIFEST_PATH}"
        )

    manifest_version = SPLIT_MANIFEST_PATH.stem

    if manifest_version != EXPECTED_SPLIT_MANIFEST_VERSION:
        raise ValueError(
            "unexpected split manifest version: "
            f"expected={EXPECTED_SPLIT_MANIFEST_VERSION}, "
            f"actual={manifest_version}"
        )


def _load_dataset_fingerprint() -> str:
    """#38에서 생성된 split summary에서 dataset fingerprint를 읽기"""
    if not DATASET_SPLIT_JSON_REPORT_PATH.is_file():
        raise FileNotFoundError(
            "dataset split summary is missing: "
            f"{DATASET_SPLIT_JSON_REPORT_PATH}"
        )

    try:
        summary = json.loads(
            DATASET_SPLIT_JSON_REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "dataset split summary is not valid JSON"
        ) from exc

    validation = summary.get("validation")

    if not isinstance(validation, dict):
        raise TypeError(
            "dataset split summary does not contain validation"
        )

    if validation.get("passed") is not True:
        raise ValueError(
            "dataset split validation did not pass"
        )

    dataset_fingerprint = summary.get(
        "dataset_fingerprint"
    )

    if not isinstance(dataset_fingerprint, str):
        raise TypeError(
            "dataset split summary does not contain "
            "a dataset fingerprint"
        )

    if len(dataset_fingerprint) != 64:
        raise ValueError(
            "dataset fingerprint must be a SHA-256 hex string"
        )

    try:
        int(dataset_fingerprint, 16)
    except ValueError as exc:
        raise ValueError(
            "dataset fingerprint must contain only "
            "hexadecimal characters"
        ) from exc

    return dataset_fingerprint


def _build_models() -> list[BasePhishingClassifier]:
    """고정된 설정을 사용하는 세 비교 모델을 생성"""
    return [
        # 기존 운영 모델과 같은 구조 피처 포함 NB를 공식 기준으로 사용
        NaiveBayesPhishingClassifier(
            include_structural_features=True,
        ),
        # 형태소 TF-IDF 기반 Logistic Regression
        LogisticRegressionPhishingClassifier(),
        # 문자 n-gram TF-IDF 기반 Linear SVM
        LinearSvmPhishingClassifier(),
    ]


def _validate_model_set(
    models: list[BasePhishingClassifier],
) -> None:
    """비교 대상 모델이 정확히 세 종류인지 검증"""
    expected_model_names = {
        "naive_bayes_structural",
        "logistic_regression_morph_tfidf",
        "linear_svm_char_tfidf",
    }

    actual_model_names = {
        model.model_name
        for model in models
    }

    if actual_model_names != expected_model_names:
        raise ValueError(
            "comparison model set does not match the expected models: "
            f"expected={sorted(expected_model_names)}, "
            f"actual={sorted(actual_model_names)}"
        )

    if len(models) != len(actual_model_names):
        raise ValueError(
            "comparison model names must be unique"
        )


def _check_artifact_targets(
    *,
    overwrite_artifacts: bool,
) -> None:
    """기존 신규 모델 artifact의 의도하지 않은 덮어쓰기를 방지"""
    if overwrite_artifacts:
        return

    existing_paths: list[Path] = []

    for artifact_directory in (
        LOGISTIC_ARTIFACT_DIRECTORY,
        LINEAR_SVM_ARTIFACT_DIRECTORY,
    ):
        model_path = artifact_directory / "model.joblib"
        metadata_path = artifact_directory / "metadata.json"

        if model_path.exists():
            existing_paths.append(model_path)

        if metadata_path.exists():
            existing_paths.append(metadata_path)

    if existing_paths:
        formatted_paths = "\n".join(
            f"- {path}"
            for path in existing_paths
        )

        raise FileExistsError(
            "comparison artifacts already exist. "
            "Use --overwrite-artifacts only when replacement "
            "is intentional:\n"
            f"{formatted_paths}"
        )


def _evaluate_models(
    *,
    models: list[BasePhishingClassifier],
    train_df,
    validation_df,
    test_df,
) -> list[ModelEvaluationResult]:
    """동일한 split에서 세 모델을 순서대로 학습하고 평가"""
    results: list[ModelEvaluationResult] = []

    for model in models:
        print(
            f"\n[Model] 학습 및 평가 시작: {model.model_name}"
        )

        result = train_and_evaluate_model(
            model,
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            target_recall=TARGET_PHISHING_RECALL,
            latency_sample_count=LATENCY_SAMPLE_COUNT,
        )

        results.append(result)

        metrics = result.test_metrics

        print(
            f"[Model] 평가 완료: {result.model_name}\n"
            f"  threshold={result.selected_threshold:.6f}\n"
            f"  precision={metrics.precision:.4f}\n"
            f"  recall={metrics.recall:.4f}\n"
            f"  f1={metrics.f1:.4f}\n"
            f"  f2={metrics.f2:.4f}\n"
            f"  TN={metrics.true_negative} "
            f"FP={metrics.false_positive} "
            f"FN={metrics.false_negative} "
            f"TP={metrics.true_positive}\n"
            f"  avg_ms={result.latency.average_ms:.3f}\n"
            f"  p95_ms={result.latency.p95_ms:.3f}"
        )

    return results


def _find_model_and_result(
    *,
    models: list[BasePhishingClassifier],
    results: list[ModelEvaluationResult],
    model_name: str,
) -> tuple[
    BasePhishingClassifier,
    ModelEvaluationResult,
]:
    """동일한 model_name의 학습 모델과 평가 결과 찾기"""
    matching_models = [
        model
        for model in models
        if model.model_name == model_name
    ]

    matching_results = [
        result
        for result in results
        if result.model_name == model_name
    ]

    if len(matching_models) != 1:
        raise RuntimeError(
            f"expected one trained model for {model_name}"
        )

    if len(matching_results) != 1:
        raise RuntimeError(
            f"expected one evaluation result for {model_name}"
        )

    return matching_models[0], matching_results[0]


def _save_new_model_artifacts(
    *,
    models: list[BasePhishingClassifier],
    results: list[ModelEvaluationResult],
    dataset_fingerprint: str,
    split_manifest_version: str,
    overwrite_artifacts: bool,
) -> None:
    """LR과 Linear SVM만 비교 artifact로 저장"""
    artifact_targets = {
        "logistic_regression_morph_tfidf": (
            LOGISTIC_ARTIFACT_DIRECTORY
        ),
        "linear_svm_char_tfidf": (
            LINEAR_SVM_ARTIFACT_DIRECTORY
        ),
    }

    for model_name, output_directory in (
        artifact_targets.items()
    ):
        model, result = _find_model_and_result(
            models=models,
            results=results,
            model_name=model_name,
        )

        save_comparison_artifact(
            model,
            threshold=result.selected_threshold,
            output_directory=output_directory,
            dataset_fingerprint=dataset_fingerprint,
            split_manifest_version=split_manifest_version,
            overwrite=overwrite_artifacts,
        )

        print(
            f"[Artifact] 저장 완료: {output_directory}"
        )


def _build_run_report(
    *,
    results: list[ModelEvaluationResult],
    dataset_fingerprint: str,
    manifest_sha256: str,
    split_manifest_version: str,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, Any]:
    """데이터와 split 문맥을 포함한 통합 JSON 보고서 생성"""
    return {
        "schema_version": (
            COMPARISON_RUN_SCHEMA_VERSION
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_policy": {
            "training_data": "train only",
            "threshold_selection_data": (
                "validation only"
            ),
            "final_metric_data": "test only",
            "target_phishing_recall": (
                TARGET_PHISHING_RECALL
            ),
            "latency_sample_count": (
                LATENCY_SAMPLE_COUNT
            ),
            "test_used_for_hyperparameter_tuning": False,
        },
        "execution_environment": _collect_execution_environment(),
        "dataset": {
            "source_path": DATA_PATH.relative_to(
                SMS_MODEL_DIR.parent
            ).as_posix(),
            "dataset_fingerprint": (
                dataset_fingerprint
            ),
        },
        "split_manifest": {
            # 전달받은 version과 path가 서로 다른 보고서를 만들지 않도록 동일한
            # 입력에서 portable path를 구성한다.
            "path": f"splits/{split_manifest_version}.csv",
            "version": split_manifest_version,
            "sha256": manifest_sha256,
            "counts": {
                "train": train_count,
                "validation": validation_count,
                "test": test_count,
            },
        },
        "models": [
            result.to_dict()
            for result in results
        ],
    }


def _render_run_markdown(
    *,
    run_report: dict[str, Any],
    results: list[ModelEvaluationResult],
) -> str:
    """실행 문맥, 평가표와 혼동행렬을 Markdown으로 렌더링"""
    split_manifest = run_report["split_manifest"]
    dataset = run_report["dataset"]
    policy = run_report["evaluation_policy"]

    lines = [
        "# Phishing Model Comparison",
        "",
        "## Reproducibility",
        "",
        f"- Dataset fingerprint: `{dataset['dataset_fingerprint']}`",
        f"- Split manifest: `{split_manifest['version']}`",
        f"- Split manifest SHA-256: `{split_manifest['sha256']}`",
        f"- Train rows: {split_manifest['counts']['train']}",
        (
            "- Validation rows: "
            f"{split_manifest['counts']['validation']}"
        ),
        f"- Test rows: {split_manifest['counts']['test']}",
        (
            "- Target phishing recall: "
            f"{policy['target_phishing_recall']:.2f}"
        ),
        "- Threshold selection: validation only",
        "- Final classification metrics: test only",
        "- Test-based hyperparameter tuning: disabled",
        "",
        "## Model Evaluation",
        "",
    ]

    # 기존 공통 Markdown 표에서 제목을 제외하고 표 부분만 재사용
    common_markdown = render_model_evaluation_markdown(
        results
    )
    common_lines = common_markdown.splitlines()

    # 첫 줄 "# Phishing Model Evaluation"과 바로 다음 빈 줄을 제외
    lines.extend(common_lines[2:])

    lines.extend(
        [
            "",
            "## Validation Threshold Selection",
            "",
            (
                "| Model | Threshold | Target Recall "
                "| Validation Recall | Target Met |"
            ),
            "|---|---:|---:|---:|:---:|",
        ]
    )

    for result in results:
        validation = result.validation

        lines.append(
            f"| {result.model_name} "
            f"| {result.selected_threshold:.6f} "
            f"| {validation.target_recall:.4f} "
            f"| {validation.recall:.4f} "
            f"| {'yes' if validation.target_recall_met else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Test Confusion Matrices",
            "",
            "| Model | TN | FP | FN | TP |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        metrics = result.test_metrics

        lines.append(
            f"| {result.model_name} "
            f"| {metrics.true_negative} "
            f"| {metrics.false_positive} "
            f"| {metrics.false_negative} "
            f"| {metrics.true_positive} |"
        )

    lines.extend(
        [
            "",
            (
                "전체 모델 및 vectorizer 설정은 "
                "`comparison_run.json`의 "
                "`models[].metadata`에서 확인할 수 있습니다."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _save_run_reports(
    *,
    run_report: dict[str, Any],
    results: list[ModelEvaluationResult],
) -> None:
    """공통 평가 보고서와 실행 문맥 보고서를 저장"""
    COMPARISON_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 기존 공통 JSON/CSV/Markdown 보고서를 생성
    save_model_evaluation_reports(
        results,
        output_directory=COMPARISON_REPORT_DIRECTORY,
    )

    comparison_json_path = (
        COMPARISON_REPORT_DIRECTORY
        / "comparison_run.json"
    )

    comparison_markdown_path = (
        COMPARISON_REPORT_DIRECTORY
        / "comparison_run.md"
    )

    comparison_json_path.write_text(
        json.dumps(
            run_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    comparison_markdown_path.write_text(
        _render_run_markdown(
            run_report=run_report,
            results=results,
        ),
        encoding="utf-8",
    )

    print(
        f"[Report] 저장 완료: "
        f"{COMPARISON_REPORT_DIRECTORY}"
    )


def run_model_comparison(
    *,
    overwrite_artifacts: bool = False,
) -> list[ModelEvaluationResult]:
    """세 모델의 전체 학습·평가·보고서·artifact 저장 흐름을 실행"""
    _require_committed_split_manifest()

    _check_artifact_targets(
        overwrite_artifacts=overwrite_artifacts
    )

    print(
        f"[Dataset] 로드 시작: {DATA_PATH}"
    )

    dataset, _unused_holdout = load_data(
        DATA_PATH
    )

    splits = split_data(
        dataset,
        create_manifest=False,
    )

    dataset_fingerprint = (
        _load_dataset_fingerprint()
    )

    split_manifest_version = (
        SPLIT_MANIFEST_PATH.stem
    )

    manifest_sha256 = _calculate_sha256(
        SPLIT_MANIFEST_PATH
    )

    print(
        "[Split] 동일 manifest 적용 완료\n"
        f"  version={split_manifest_version}\n"
        f"  sha256={manifest_sha256}\n"
        f"  train={len(splits.train)}\n"
        f"  validation={len(splits.validation)}\n"
        f"  test={len(splits.test)}"
    )

    models = _build_models()
    _validate_model_set(models)

    results = _evaluate_models(
        models=models,
        train_df=splits.train,
        validation_df=splits.validation,
        test_df=splits.test,
    )

    run_report = _build_run_report(
        results=results,
        dataset_fingerprint=dataset_fingerprint,
        manifest_sha256=manifest_sha256,
        split_manifest_version=split_manifest_version,
        train_count=len(splits.train),
        validation_count=len(splits.validation),
        test_count=len(splits.test),
    )

    _save_run_reports(
        run_report=run_report,
        results=results,
    )

    _save_new_model_artifacts(
        models=models,
        results=results,
        dataset_fingerprint=dataset_fingerprint,
        split_manifest_version=split_manifest_version,
        overwrite_artifacts=overwrite_artifacts,
    )

    return results


def _build_argument_parser() -> argparse.ArgumentParser:
    """명령행 인자를 구성"""
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare NB, morphological Logistic "
            "Regression, and character Linear SVM models."
        )
    )

    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
        help=(
            "Replace existing Logistic Regression and "
            "Linear SVM comparison artifacts."
        ),
    )

    return parser


def main() -> None:
    """명령행 실행 진입점"""
    parser = _build_argument_parser()
    arguments = parser.parse_args()

    results = run_model_comparison(
        overwrite_artifacts=(
            arguments.overwrite_artifacts
        )
    )

    print("\n[Comparison Summary]")

    for result in results:
        metrics = result.test_metrics

        print(
            f"- {result.model_name}: "
            f"recall={metrics.recall:.4f}, "
            f"f2={metrics.f2:.4f}, "
            f"fn={metrics.false_negative}, "
            f"avg_ms={result.latency.average_ms:.3f}"
        )


if __name__ == "__main__":
    main()
