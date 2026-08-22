"""SMS 모델 공통 평가 API"""

from .adoption import (
    AdoptionAssessment,
    AdoptionCriteria,
    AdoptionVerdict,
    CriterionResult,
    evaluate_adoption,
)
from .evaluator import (
    ModelEvaluationResult,
    train_and_evaluate_model,
)
from .latency import (
    LatencyMetrics,
    measure_single_inference_latency,
)
from .metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from .reporting import (
    save_model_evaluation_reports,
)
from .standalone_bands import (
    BandEdges,
    BandEdgesUnreachableError,
    assign_bands,
    measure_edge_transfer,
    measure_reliability,
    select_standalone_bands,
    summarize_bands,
    sweep_band_frontier,
)
from .threshold import (
    ProbabilityThresholdSelection,
    ThresholdInfeasibleError,
    ThresholdSelection,
    select_probability_threshold,
    select_probability_threshold_with_fallback,
    select_validation_threshold,
)

__all__ = [
    "AdoptionAssessment",
    "AdoptionCriteria",
    "AdoptionVerdict",
    "BandEdges",
    "BandEdgesUnreachableError",
    "ClassificationMetrics",
    "CriterionResult",
    "LatencyMetrics",
    "ModelEvaluationResult",
    "ProbabilityThresholdSelection",
    "ThresholdInfeasibleError",
    "ThresholdSelection",
    "assign_bands",
    "calculate_classification_metrics",
    "evaluate_adoption",
    "measure_edge_transfer",
    "measure_reliability",
    "measure_single_inference_latency",
    "save_model_evaluation_reports",
    "select_probability_threshold",
    "select_probability_threshold_with_fallback",
    "select_standalone_bands",
    "select_validation_threshold",
    "summarize_bands",
    "sweep_band_frontier",
    "train_and_evaluate_model",
]
