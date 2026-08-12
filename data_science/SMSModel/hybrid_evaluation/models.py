"""하이브리드 텍스트 분석 평가에서 사용하는 데이터 계약"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class EvaluationMode(str, Enum):
    """비교할 세 가지 평가 모드"""

    # Stacking 자체 모델만 사용
    SELF_MODEL_ONLY = "SELF_MODEL_ONLY"

    # 모든 메시지를 LLM으로 분석
    LLM_ONLY = "LLM_ONLY"

    # Stacking이 불확실한 메시지만 LLM으로 재검증
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class ClassificationMetrics:
    """피싱을 positive class로 계산한 이진 분류 성능"""

    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float

    # 혼동 행렬 값
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 가능한 dictionary로 변환"""

        return asdict(self)


@dataclass(frozen=True)
class LatencyMetrics:
    """메시지 단위 end-to-end latency 집계"""

    sample_count: int
    average_ms: float
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 가능한 dictionary로 변환합니다."""

        return asdict(self)


@dataclass(frozen=True)
class TokenUsage:
    """한 번의 logical LLM 호출에서 확인된 token usage"""
    # Bedrock 요청이 실패하거나 응답에 usage가 없으면 두 값 모두 None으로 설정
    # 한쪽 값만 있는 불완전한 usage는 비용 계산에서 제외

    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class CostMetrics:
    """LLM token usage를 기준으로 계산한 비용 통계"""

    input_tokens: int
    output_tokens: int

    # input/output token을 모두 확인할 수 있었던 호출 수
    measured_call_count: int

    # 실패 또는 usage 누락으로 비용을 계산하지 못한 호출 수
    unmeasured_call_count: int

    total_cost: float
    cost_per_message: float

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 가능한 dictionary로 변환합니다."""

        return asdict(self)


@dataclass(frozen=True)
class OperationalOutcome:
    """메시지 한 건의 LLM 호출 및 fallback 결과"""

    # 실제로 LLM 분석기를 호출했는지
    llm_called: bool

    # LLM 결과를 최종 판정에 사용할 수 있었는지
    llm_available: bool

    # LLM 실패 후 자체 모델 결과를 사용했는지
    fallback_applied: bool

    # 자체 모델과 LLM이 모두 실패했는지
    all_engines_unavailable: bool


@dataclass(frozen=True)
class OperationalMetrics:
    """평가 모드의 호출률 및 장애 경로 집계"""

    total_message_count: int
    llm_call_count: int
    llm_call_rate: float
    llm_success_count: int
    llm_failure_count: int
    fallback_count: int
    all_engines_unavailable_count: int

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화가 가능한 dictionary로 변환"""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    """메시지별 비민감 결과"""

    # 복원할 수 없는 익명 식별자
    sample_id: str

    mode: EvaluationMode
    expected_label: str
    predicted_label: str

    # 요청 전체 처리 시간
    latency_ms: float

    # 텍스트 판정 엔진의 사용 가능 여부
    result_available: bool

    # LLM 호출 및 fallback 관측값
    llm_called: bool
    llm_available: bool
    fallback_applied: bool
    all_engines_unavailable: bool

    # 운영 응답에서 사용하는 machine-readable 값만 허용
    decision_source: str
    routing_decision: str | None
    routing_reason: str | None
    error_code: str | None

    # 공급자와 모델 식별자는 재현성을 위해 저장
    llm_provider: str | None
    llm_model: str | None

    # 실제 응답에서 제공된 token usage
    input_tokens: int | None
    output_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        """JSON/CSV 저장이 가능한 dictionary로 변환"""

        payload = asdict(self)

        # Enum은 문자열 값으로 직렬화
        payload["mode"] = self.mode.value

        return payload
