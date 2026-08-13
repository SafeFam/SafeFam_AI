"""원문을 저장하지 않는 Claude test 예측 캐시"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.analysis.text.llm_analyzer import SYSTEM_PROMPT

CACHE_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
CACHE_SPLIT = "test"
CACHE_PROVIDER = "AWS_BEDROCK"
PROMPT_NAME = "smishing-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ALLOWED_GRADES = {"SAFE", "SUSPICIOUS", "DANGEROUS", "UNKNOWN"}


def _validate_text_fingerprint(value: object) -> str:
    """원문 대신 lowercase SHA-256 식별자만 허용합니다."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("text_fingerprint must be a lowercase SHA-256 digest")
    return value

def build_prompt_version() -> str:
    """프롬프트 이름과 실제 내용 fingerprint를 결합"""
    digest = hashlib.sha256(
        SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()

    return f"{PROMPT_NAME}:{digest}"

def calculate_dataset_fingerprint(
    rows: Iterable[tuple[str, str]],   
) -> str:
    """test fingerprint와 label로 평가 데이터셋을 식별"""

    normalized = sorted(
        (_validate_text_fingerprint(fingerprint), str(label))
        for fingerprint, label in rows
    )

    if not normalized:
        raise ValueError(
            "cannot fingerprint an empty dataset"
        )

    if len({fingerprint for fingerprint, _ in normalized}) != len(
        normalized
    ):
        raise ValueError(
            "dataset fingerprints must be unique"
        )

    digest = hashlib.sha256()

    for fingerprint, label in normalized:
        if label not in {"normal", "phishing"}:
            raise ValueError(
                f"unsupported label: {label}"
            )

        digest.update(fingerprint.encode("utf-8"))
        digest.update(b"\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def _optional_token_count(
    value: object,
) -> int | None:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return None

def _atomic_write_json(
        path: Path,
        payload: dict[str, Any],
) -> None:
    """중단 중 기존 캐시가 손상되지 않도록 원자적으로 저장"""

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

class ClaudeTestCache:
    """test split에만 사용하는 Claude 결과 캐시"""

    def __init__(
        self,
        *,
        path: Path,
        dataset_fingerprint: str,
        model_id: str,
        region: str,
        prompt_version: str,
    ) -> None:
        self.path = Path(path)
        self.dataset_fingerprint = dataset_fingerprint
        self.model_id = model_id
        self.region = region
        self.prompt_version = prompt_version
        self._entries: dict[str, dict[str, Any]] = {}

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """외부 변경을 막기 위해 복사본 반환"""

        return deepcopy(self._entries)

    def get_entry(self, text_fingerprint: str) -> dict[str, Any] | None:
        """Return one internal cache entry for read-only evaluation use."""
        return self._entries.get(_validate_text_fingerprint(text_fingerprint))

    def load(self) -> None:
        """기존 캐시를 읽고 모든 재현성 조건을 검증"""

        if not self.path.is_file():
            self._entries = {}
            return

        payload = json.loads(
            self.path.read_text(encoding="utf-8")
        )

        expected_metadata = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "source_split": CACHE_SPLIT,
            "provider": CACHE_PROVIDER,
            "model_id": self.model_id,
            "region": self.region,
            "prompt_version": self.prompt_version,
            "dataset_fingerprint": self.dataset_fingerprint,
        }

        for field_name, expected_value in expected_metadata.items():
            actual_value = payload.get(field_name)

            if actual_value != expected_value:
                raise ValueError(
                    "Claude test cache metadata mismatch: "
                    f"{field_name}; "
                    f"expected={expected_value!r}, "
                    f"actual={actual_value!r}"
                )

        entries = payload.get("predictions")

        if not isinstance(entries, list):
            raise ValueError(
                "Claude test cache predictions must be a list"
            )

        loaded: dict[str, dict[str, Any]] = {}

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Claude test cache entry must be an object"
                )

            fingerprint = _validate_text_fingerprint(
                entry.get("text_fingerprint")
            )

            if fingerprint in loaded:
                raise ValueError(
                    "duplicate fingerprint in Claude test cache"
                )

            # 고정된 필드만 허용
            allowed_fields = {
                "text_fingerprint",
                "available",
                "grade",
                "risk_score",
                "error_code",
                "input_tokens",
                "output_tokens",
                "latency_ms",
            }

            unexpected = set(entry) - allowed_fields

            if unexpected:
                raise ValueError(
                    "unexpected fields in Claude test cache: "
                    f"{sorted(unexpected)}"
                )

            _validate_cache_entry(entry)

            loaded[fingerprint] = entry

        self._entries = loaded

    def save(self) -> None:
        """원문이나 자격 증명 없이 현재 캐시를 저장"""
        _atomic_write_json(
            self.path,
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "evaluation_schema_version": (
                    EVALUATION_SCHEMA_VERSION
                ),
                "source_split": CACHE_SPLIT,
                "provider": CACHE_PROVIDER,
                "model_id": self.model_id,
                "region": self.region,
                "prompt_version": self.prompt_version,
                "dataset_fingerprint": (
                    self.dataset_fingerprint
                ),
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "predictions": [
                    self._entries[fingerprint]
                    for fingerprint in sorted(self._entries)
                ],
            },
        )

    def contains(
        self,
        text_fingerprint: str,
        *,
        require_success: bool = False,
    ) -> bool:
        entry = self._entries.get(text_fingerprint)

        if entry is None:
            return False

        if require_success:
            return entry.get("available") is True

        return True

    def store_analysis(
        self,
        *,
        text_fingerprint: str,
        analysis: dict[str, Any],
    ) -> None:
        """LLM 응답에서 평가에 필요한 안전한 필드만 추출"""
        text_fingerprint = _validate_text_fingerprint(text_fingerprint)

        if not isinstance(analysis, dict):
            raise TypeError("analysis must be a dictionary")

        result = analysis.get("result") or {}
        usage = analysis.get("usage") or {}

        if not isinstance(result, dict) or not isinstance(usage, dict):
            raise TypeError("LLM result and usage must be dictionaries")

        provider = analysis.get("provider")
        model_id = analysis.get("model_id")

        if provider not in {None, CACHE_PROVIDER}:
            raise ValueError(
                "LLM response provider does not match cache provider"
            )

        if model_id not in {None, self.model_id}:
            raise ValueError(
                "LLM response model does not match cache model"
            )

        grade = result.get("grade")
        risk_score = result.get("risk_score")
        error_code = result.get("error_message")

        available = (
            grade in {"SAFE", "SUSPICIOUS", "DANGEROUS"}
            and isinstance(risk_score, int)
            and not isinstance(risk_score, bool)
            and 0 <= risk_score <= 100
            and not error_code
        )

        latency_ms = analysis.get("latency_ms")

        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not math.isfinite(float(latency_ms))
            or latency_ms < 0
        ):
            latency_ms = None
        else:
            latency_ms = float(latency_ms)

        self._entries[text_fingerprint] = {
            "text_fingerprint": text_fingerprint,
            "available": available,
            "grade": grade if available else "UNKNOWN",
            "risk_score": risk_score if available else None,
            "error_code": (
                None if available else _safe_error_code(error_code)
            ),
            "input_tokens": _optional_token_count(
                usage.get("input_tokens")
            ),
            "output_tokens": _optional_token_count(
                usage.get("output_tokens")
            ),
            "latency_ms": latency_ms,
        }

    def require_complete(
        self,
        expected_fingerprints: set[str],
    ) -> None:
        """offline 평가 전에 모든 test 샘플이 존재하는지 확인"""
        cached_fingerprints = set(self._entries)

        missing = expected_fingerprints - cached_fingerprints
        unexpected = cached_fingerprints - expected_fingerprints

        if missing:
            raise RuntimeError(
                "Claude test cache is incomplete: "
                f"{len(missing)} predictions are missing"
            )

        if unexpected:
            raise RuntimeError(
                "Claude test cache contains predictions outside "
                f"the current test split: {len(unexpected)}"
            )

    async def analyze_cached(
        self,
        text_fingerprint: str,
    ) -> dict[str, Any]:
        """LLM 분석 응답 형태로 복원"""
        entry = self._entries.get(text_fingerprint)

        if entry is None:
            raise KeyError(
                "Claude prediction is missing from test cache"
            )

        available = entry.get("available") is True

        return {
            "is_mock": False,
            "is_cached": True,
            "provider": CACHE_PROVIDER,
            "model_id": self.model_id,
            "usage": {
                "input_tokens": entry.get("input_tokens"),
                "output_tokens": entry.get("output_tokens"),
            },
            "latency_ms": entry.get("latency_ms"),
            "result": {
                "grade": (
                    entry.get("grade")
                    if available
                    else "UNKNOWN"
                ),
                "risk_score": (
                    entry.get("risk_score")
                    if available
                    else None
                ),
                "tone_analysis": "",
                "evidence": [],
                "reason": (
                    "Loaded from the Claude test cache."
                    if available
                    else "Cached Claude analysis is unavailable."
                ),
                "error_message": (
                    None
                    if available
                    else entry.get("error_code")
                ),
            },
        }


def _safe_error_code(value: object) -> str:
    """상세 예외 문자열 대신 허용된 machine-readable 코드만 저장"""
    allowed = {
        "LLM_TIMEOUT",
        "LLM_THROTTLED",
        "LLM_PROVIDER_ERROR",
        "LLM_INVALID_RESPONSE",
        "LLM_ANALYZER_FAILED",
    }

    if isinstance(value, str) and value in allowed:
        return value

    return "LLM_TEST_EVALUATION_FAILED"


def _validate_cache_entry(entry: dict[str, Any]) -> None:
    """로드한 캐시 값의 타입과 범위를 검증합니다."""
    available = entry.get("available")
    if not isinstance(available, bool):
        raise ValueError("cached available must be a boolean")

    grade = entry.get("grade")
    if grade not in _ALLOWED_GRADES:
        raise ValueError("cached grade is invalid")

    risk_score = entry.get("risk_score")
    error_code = entry.get("error_code")
    if available:
        if grade == "UNKNOWN":
            raise ValueError("available cache entry cannot have UNKNOWN grade")
        if (
            isinstance(risk_score, bool)
            or not isinstance(risk_score, int)
            or not 0 <= risk_score <= 100
        ):
            raise ValueError("available cache risk_score is invalid")
        if error_code is not None:
            raise ValueError("available cache entry cannot have an error_code")
    else:
        if grade != "UNKNOWN" or risk_score is not None:
            raise ValueError("unavailable cache entry must use UNKNOWN and null score")
        if not isinstance(error_code, str) or not error_code:
            raise ValueError("unavailable cache entry requires an error_code")

    for field_name in ("input_tokens", "output_tokens"):
        value = entry.get(field_name)
        if value is not None and _optional_token_count(value) is None:
            raise ValueError(f"cached {field_name} is invalid")

    latency_ms = entry.get("latency_ms")
    if latency_ms is not None and (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(float(latency_ms))
        or latency_ms < 0
    ):
        raise ValueError("cached latency_ms is invalid")
