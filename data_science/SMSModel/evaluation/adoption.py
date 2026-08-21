"""Stacking 단독 운영 채택 기준"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class AdoptionVerdict(str, Enum):
    """artifact 하나에 대한 종합 판정"""

    PASS = PASS
    FAIL = FAIL
    INSUFFICIENT_EVIDENCE = INSUFFICIENT_EVIDENCE


@dataclass(frozen=True)
class AdoptionCriteria:
    """단독 운영 승격에 필요한 최소 조건"""

    # 최소 '의심' 이상으로 걸러내야 하는 피싱 비율.
    # 현행 하이브리드가 놓치는 피싱이 0건이라 완화 근거가 없다 (#100 §2.3)
    min_coverage_recall: float = 0.95

    # 자동 경고가 잘못 울릴 정상 비율. 하루 15건 기준 주 1회 오경보
    max_alert_false_positive_rate: float = 0.01

    # '의심' 카드가 뜨는 정상 비율. 하루 15건 기준 하루 2건 (#100 §2.2)
    max_uncertain_normal_share: float = 0.133

    # 단건 추론 P95 지연시간
    max_p95_latency_ms: float = 50.0

    # 오탐률 상한을 오탐 0건으로 확인하려면 rule of three로 3/n <= 상한
    min_normal_samples: int = 300

    # 놓침 5% 상한을 0건으로 확인하려면 3/n <= 0.05
    min_phishing_samples: int = 60


@dataclass(frozen=True)
class CriterionResult:
    """기준 하나에 대한 판정"""

    name: str
    outcome: str
    measured: float | None
    required: float
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """리포트에 기록할 dictionary로 변환"""
        return asdict(self)


@dataclass(frozen=True)
class AdoptionAssessment:
    """artifact 하나에 대한 판정 결과 전체"""

    verdict: str
    criteria: list[CriterionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """리포트에 기록할 dictionary로 변환"""
        return {
            "verdict": self.verdict,
            "criteria": [
                criterion.to_dict() for criterion in self.criteria
            ],
        }

    @property
    def failed(self) -> list[CriterionResult]:
        """기준에 미달한 항목"""
        return [c for c in self.criteria if c.outcome == FAIL]

    @property
    def unmeasurable(self) -> list[CriterionResult]:
        """표본이 부족해 판정할 수 없는 항목"""
        return [c for c in self.criteria if c.outcome == INSUFFICIENT_EVIDENCE]


def _at_most(
    name: str,
    measured: float,
    limit: float,
    *,
    detail: str = "",
) -> CriterionResult:
    """상한 기준 하나를 판정"""
    return CriterionResult(
        name=name,
        outcome=PASS if measured <= limit else FAIL,
        measured=float(measured),
        required=float(limit),
        detail=detail,
    )


def _at_least(
    name: str,
    measured: float,
    limit: float,
    *,
    detail: str = "",
) -> CriterionResult:
    """하한 기준 하나를 판정"""
    return CriterionResult(
        name=name,
        outcome=PASS if measured >= limit else FAIL,
        measured=float(measured),
        required=float(limit),
        detail=detail,
    )


def _sample_size(
    name: str,
    measured: int,
    required: int,
    *,
    detail: str,
) -> CriterionResult:
    """표본 수가 판정에 충분한지 확인"""
    return CriterionResult(
        name=name,
        outcome=PASS if measured >= required else INSUFFICIENT_EVIDENCE,
        measured=float(measured),
        required=float(required),
        detail=detail,
    )


def evaluate_adoption(
    band_summary: dict,
    *,
    p95_latency_ms: float,
    criteria: AdoptionCriteria | None = None,
) -> AdoptionAssessment:
    """실제 문자 평가셋 측정치를 기준과 대조해 판정"""
    criteria = criteria or AdoptionCriteria()
    by_band = band_summary["by_band"]

    normal_count = int(band_summary["normal_count"])
    phishing_count = int(band_summary["phishing_count"])
    uncertain_normal_share = by_band["uncertain"]["normal_share"]

    results = [
        _sample_size(
            "normal_sample_size",
            normal_count,
            criteria.min_normal_samples,
            detail=(
                "오탐 0건이 상한 준수의 증거가 되려면 정상 표본이 필요하다"
            ),
        ),
        _sample_size(
            "phishing_sample_size",
            phishing_count,
            criteria.min_phishing_samples,
            detail="놓침 상한을 확인하는 데 필요한 피싱 표본",
        ),
        _at_least(
            "coverage_recall",
            band_summary["coverage_recall"],
            criteria.min_coverage_recall,
            detail="최소 '의심' 이상으로 걸러낸 피싱 비율",
        ),
        _at_most(
            "alert_false_positive_rate",
            band_summary["alert_false_positive_rate"],
            criteria.max_alert_false_positive_rate,
            detail="자동 경고가 잘못 울린 정상 비율",
        ),
        _at_most(
            "uncertain_normal_share",
            uncertain_normal_share,
            criteria.max_uncertain_normal_share,
            detail="'의심' 카드가 뜬 정상 비율",
        ),
        _at_most(
            "p95_latency_ms",
            p95_latency_ms,
            criteria.max_p95_latency_ms,
            detail="단건 추론 P95 지연시간",
        ),
    ]

    if any(result.outcome == FAIL for result in results):
        verdict = FAIL
    elif any(result.outcome == INSUFFICIENT_EVIDENCE for result in results):
        verdict = INSUFFICIENT_EVIDENCE
    else:
        verdict = PASS

    return AdoptionAssessment(verdict=verdict, criteria=results)
