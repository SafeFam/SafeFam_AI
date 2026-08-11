"""Validation 데이터로 하이브리드 Gemini 호출 구간을 선정"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.analysis.text.gemini_analyzer import (
    GEMINI_API_KEY,
    MOCK_ENABLED,
    analyze_text_with_gemini,
)
from data_science.SMSModel.modeling.hybrid_thresholds import (
    HybridThresholdSelection,
    select_hybrid_thresholds,
)
from data_science.SMSModel.modeling.stacking import (
    StackingPhishingClassifier,
)
from data_science.SMSModel.train_sms import (
    DATA_PATH,
    SPLIT_MANIFEST_PATH,
    load_data,
    split_data,
)


SMS_MODEL_DIRECTORY = Path(__file__).resolve().parent

STACKING_ARTIFACT_DIRECTORY = (
    SMS_MODEL_DIRECTORY
    / "artifacts"
    / "stacking"
)

STACKING_MODEL_PATH = (
    STACKING_ARTIFACT_DIRECTORY
    / "model.joblib"
)

STACKING_METADATA_PATH = (
    STACKING_ARTIFACT_DIRECTORY
    / "metadata.json"
)

# 원문 없이 fingerprint와 Gemini 점수만 저장
GEMINI_VALIDATION_CACHE_PATH = (
    STACKING_ARTIFACT_DIRECTORY
    / "gemini_validation_predictions.json"
)

# 사람이 확인할 수 있는 결과 보고서
HYBRID_POLICY_REPORT_PATH = (
    STACKING_ARTIFACT_DIRECTORY
    / "hybrid_policy.json"
)

def _atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    """중간 실패 시 기존 파일이 손상되지 않도록 원자적으로 저장"""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)

def _calculate_sha256_bytes(
    data: bytes,
) -> str:
    """메모리에 로드한 artifact bytes의 SHA-256 계산"""

    return hashlib.sha256(data).hexdigest()

def _load_stacking_classifier(
) -> StackingPhishingClassifier:
    """Checksum을 검증한 뒤 stacking classifier를 로드"""

    if not STACKING_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"stacking model is missing: "
            f"{STACKING_MODEL_PATH}"
        )

    if not STACKING_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"stacking metadata is missing: "
            f"{STACKING_METADATA_PATH}"
        )

    metadata = json.loads(
        STACKING_METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    expected_digest = metadata.get(
        "model_sha256"
    )

    if not isinstance(expected_digest, str):
        raise ValueError(
            "stacking metadata does not contain model_sha256"
        )

    artifact_bytes = (
        STACKING_MODEL_PATH.read_bytes()
    )

    actual_digest = _calculate_sha256_bytes(
        artifact_bytes
    )

    if actual_digest != expected_digest:
        raise ValueError(
            "stacking artifact checksum mismatch"
        )

    # 검증한 동일한 bytes를 역직렬화
    artifact_buffer = io.BytesIO(
        artifact_bytes
    )

    payload = joblib.load(
        artifact_buffer
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "stacking artifact must be a dictionary"
        )

    if payload.get("schema_version") != 1:
        raise ValueError(
            "unsupported stacking artifact schema"
        )

    classifier = payload.get(
        "classifier"
    )

    if not isinstance(
        classifier,
        StackingPhishingClassifier,
    ):
        raise TypeError(
            "stacking classifier is missing"
        )

    return classifier

def _load_validation_data():
    """기존 split manifest로 validation 데이터만 로드"""

    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"split manifest is required: "
            f"{SPLIT_MANIFEST_PATH}"
        )

    dataset, _unused_holdout = load_data(
        DATA_PATH
    )

    splits = split_data(
        dataset,
        create_manifest=False,
    )

    validation = splits.validation.copy()

    required_columns = {
        "text",
        "label",
        "text_fingerprint",
    }

    missing = (
        required_columns - set(validation.columns)
    )

    if missing:
        raise ValueError(
            f"validation columns are missing: {missing}"
        )

    if validation[
        "text_fingerprint"
    ].duplicated().any():
        raise ValueError(
            "validation fingerprints must be unique"
        )

    return validation

def _load_cached_predictions(
) -> dict[str, dict[str, Any]]:
    """기존 Gemini validation cache를 fingerprint 기준으로 로드"""

    if not GEMINI_VALIDATION_CACHE_PATH.is_file():
        return {}

    payload = json.loads(
        GEMINI_VALIDATION_CACHE_PATH.read_text(
            encoding="utf-8"
        )
    )

    if payload.get("schema_version") != 1:
        raise ValueError(
            "unsupported Gemini validation cache schema"
        )

    predictions = (
        payload.get("predictions") or []
    )

    cached: dict[str, dict[str, Any]] = {}

    for prediction in predictions:
        fingerprint = prediction.get(
            "text_fingerprint"
        )

        if not isinstance(fingerprint, str):
            raise ValueError(
                "cached prediction fingerprint is invalid"
            )

        if fingerprint in cached:
            raise ValueError(
                "duplicate fingerprint in Gemini cache"
            )

        cached[fingerprint] = prediction

    return cached

def _save_cached_predictions(
    cached: dict[str, dict[str, Any]],
) -> None:
    """원문 없이 Gemini 예측 점수만 저장"""

    _atomic_write_json(
        GEMINI_VALIDATION_CACHE_PATH,
        {
            "schema_version": 1,
            "updated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "predictions": [
                cached[fingerprint]
                for fingerprint in sorted(cached)
            ],
        },
    )

async def collect_gemini_predictions(
    validation,
    *,
    delay_seconds: float,
) -> None:
    """Validation 메시지의 Gemini 점수를 수집"""

    if MOCK_ENABLED:
        raise RuntimeError(
            "MOCK_SECURITY_API must be false "
            "during threshold selection"
        )

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is required "
            "to collect validation predictions"
        )

    if delay_seconds < 0:
        raise ValueError(
            "delay_seconds must not be negative"
        )

    cached = _load_cached_predictions()

    total = len(validation)

    for position, row in enumerate(
        validation.itertuples(index=False),
        start=1,
    ):
        fingerprint = str(
            row.text_fingerprint
        )

        previous = cached.get(
            fingerprint
        )

        # 이전 호출이 성공했다면 비용 중복 지출 X
        if (
            previous is not None
            and previous.get("available") is True
        ):
            print(
                f"[Gemini] skip cached "
                f"{position}/{total}"
            )
            continue

        analysis = await analyze_text_with_gemini(
            str(row.text)
        )

        result = (
            analysis.get("result") or {}
        )

        score = result.get("risk_score")
        error_message = result.get(
            "error_message"
        )
        grade = result.get("grade")

        available = (
            isinstance(score, int)
            and 0 <= score <= 100
            and grade != "UNKNOWN"
            and not error_message
        )

        cached[fingerprint] = {
            "text_fingerprint": fingerprint,
            "available": available,
            "risk_score": (
                score if available else None
            ),
            # 구체적인 예외 메시지나 원문은 저장 X
            "error_code": (
                None
                if available
                else "GEMINI_VALIDATION_FAILED"
            ),
        }

        # 매 요청 후 저장하여 중간 중단 후 재개할 수 있게 함
        _save_cached_predictions(
            cached
        )

        print(
            f"[Gemini] collected "
            f"{position}/{total} "
            f"available={available}"
        )

        if delay_seconds > 0:
            await asyncio.sleep(
                delay_seconds
            )

def _ordered_gemini_scores(
    validation,
) -> np.ndarray:
    """Validation 순서와 일치하는 Gemini 점수 배열 생성"""

    cached = _load_cached_predictions()

    expected_fingerprints = {
        str(value)
        for value in validation[
            "text_fingerprint"
        ]
    }

    missing = (
        expected_fingerprints - set(cached)
    )

    if missing:
        raise RuntimeError(
            f"Gemini predictions are missing: "
            f"{len(missing)} rows"
        )

    failed = [
        fingerprint
        for fingerprint in expected_fingerprints
        if cached[fingerprint].get(
            "available"
        ) is not True
    ]

    if failed:
        raise RuntimeError(
            f"Gemini predictions failed: "
            f"{len(failed)} rows; "
            "run collection again"
        )

    scores = [
        cached[
            str(fingerprint)
        ]["risk_score"]
        for fingerprint in validation[
            "text_fingerprint"
        ]
    ]

    return np.asarray(
        scores,
        dtype=np.float64,
    )

def _write_policy_report(
    selection: HybridThresholdSelection,
) -> None:
    """선정 결과를 별도 정책 artifact에 저장"""

    payload = {
        "schema_version": 1,
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_split": "validation",
        "split_manifest": (
            SPLIT_MANIFEST_PATH.name
        ),
        "gemini_phishing_score": 40,
        "selection": selection.to_dict(),
    }

    _atomic_write_json(
        HYBRID_POLICY_REPORT_PATH,
        payload,
    )

def _update_stacking_metadata(
    selection: HybridThresholdSelection,
) -> None:
    """기존 stacking metadata에 hybrid policy를 추가"""

    metadata = json.loads(
        STACKING_METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    metadata["hybrid_policy"] = (
        selection.to_dict()
    )

    _atomic_write_json(
        STACKING_METADATA_PATH,
        metadata,
    )

def select_and_save_thresholds(
    validation,
    *,
    target_recall: float,
) -> HybridThresholdSelection:
    """Stacking/Gemini validation 결과로 임계값을 선정하고 저장"""

    classifier = (
        _load_stacking_classifier()
    )

    (
        stacking_probabilities,
        unavailable_models,
    ) = classifier.predict_probabilities(
        validation
    )

    if unavailable_models:
        raise RuntimeError(
            "all stacking base models must be available "
            "during hybrid threshold selection: "
            f"{unavailable_models}"
        )

    gemini_scores = (
        _ordered_gemini_scores(
            validation
        )
    )

    selection = select_hybrid_thresholds(
        stacking_probabilities=(
            stacking_probabilities
        ),
        gemini_scores=gemini_scores,
        labels=validation[
            "label"
        ].astype(str).to_numpy(),
        target_recall=target_recall,

        gemini_phishing_score=40,
    )

    _write_policy_report(
        selection
    )

    _update_stacking_metadata(
        selection
    )

    return selection

async def async_main(
    *,
    collect_gemini: bool,
    delay_seconds: float,
    target_recall: float,
) -> None:
    validation = _load_validation_data()

    if collect_gemini:
        await collect_gemini_predictions(
            validation,
            delay_seconds=delay_seconds,
        )

    selection = select_and_save_thresholds(
        validation,
        target_recall=target_recall,
    )

    print(
        "[Hybrid Threshold] selection completed"
    )

    print(
        "  normal_probability_max="
        f"{selection.normal_probability_max:.8f}"
    )

    print(
        "  phishing_probability_min="
        f"{selection.phishing_probability_min:.8f}"
    )

    print(
        f"  precision={selection.precision:.4f}"
    )

    print(
        f"  recall={selection.recall:.4f}"
    )

    print(
        f"  f2={selection.f2:.4f}"
    )

    print(
        "  gemini_call_rate="
        f"{selection.gemini_call_rate:.4f}"
    )

    print(
        "  gemini_calls="
        f"{selection.gemini_call_count}/"
        f"{selection.validation_count}"
    )

    print()
    print(
        "Apply these runtime settings:"
    )

    print(
        "STACKING_NORMAL_PROBABILITY_MAX="
        f"{selection.normal_probability_max:.8f}"
    )

    print(
        "STACKING_PHISHING_PROBABILITY_MIN="
        f"{selection.phishing_probability_min:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select conditional Gemini thresholds "
            "using validation predictions."
        )
    )

    parser.add_argument(
        "--collect-gemini",
        action="store_true",
        help=(
            "Call Gemini for validation rows "
            "before selecting thresholds."
        ),
    )

    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help=(
            "Delay between Gemini validation calls."
        ),
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
    )

    arguments = parser.parse_args()

    asyncio.run(
        async_main(
            collect_gemini=(
                arguments.collect_gemini
            ),
            delay_seconds=(
                arguments.delay_seconds
            ),
            target_recall=(
                arguments.target_recall
            ),
        )
    )


if __name__ == "__main__":
    main()