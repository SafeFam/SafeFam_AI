"""단독 운영 채택 기준 판정 테스트"""

import pytest

from data_science.SMSModel.evaluation.adoption import (
    FAIL,
    INSUFFICIENT_EVIDENCE,
    PASS,
    AdoptionCriteria,
    evaluate_adoption,
)


def build_band_summary(
    *,
    coverage_recall: float = 0.98,
    alert_false_positive_rate: float = 0.005,
    uncertain_normal_share: float = 0.05,
    normal_count: int = 400,
    phishing_count: int = 200,
) -> dict:
    """기준을 전부 만족하는 측정치를 기본으로 하고 필요한 항목만 흔든다"""
    return {
        "normal_count": normal_count,
        "phishing_count": phishing_count,
        "coverage_recall": coverage_recall,
        "alert_false_positive_rate": alert_false_positive_rate,
        "by_band": {
            "uncertain": {"normal_share": uncertain_normal_share},
        },
    }


def test_passes_when_every_criterion_is_met() -> None:
    """모든 기준을 만족하면 통과한다"""
    assessment = evaluate_adoption(
        build_band_summary(),
        p95_latency_ms=10.0,
    )

    assert assessment.verdict == PASS
    assert assessment.failed == []
    assert assessment.unmeasurable == []


def test_small_normal_sample_cannot_prove_the_ceiling() -> None:
    """정상 표본이 부족하면 통과가 아니라 판정불가다

    오탐 0건은 표본이 작을수록 쉽게 나온다. 이 구분이 없으면 데이터가
    적을수록 기준을 통과하기 쉬워진다.
    """
    assessment = evaluate_adoption(
        build_band_summary(normal_count=74, alert_false_positive_rate=0.0),
        p95_latency_ms=10.0,
    )

    assert assessment.verdict == INSUFFICIENT_EVIDENCE
    assert [c.name for c in assessment.unmeasurable] == ["normal_sample_size"]
    assert assessment.failed == []


def test_definite_failure_outranks_missing_evidence() -> None:
    """확실한 미달이 있으면 표본을 더 모아도 결론이 같으므로 FAIL이다"""
    assessment = evaluate_adoption(
        build_band_summary(
            normal_count=74,
            alert_false_positive_rate=0.20,
        ),
        p95_latency_ms=10.0,
    )

    assert assessment.verdict == FAIL


@pytest.mark.parametrize(
    ("kwargs", "latency", "expected_failure"),
    [
        ({"coverage_recall": 0.90}, 10.0, "coverage_recall"),
        (
            {"alert_false_positive_rate": 0.05},
            10.0,
            "alert_false_positive_rate",
        ),
        ({"uncertain_normal_share": 0.30}, 10.0, "uncertain_normal_share"),
        ({}, 120.0, "p95_latency_ms"),
    ],
)
def test_each_criterion_can_fail_on_its_own(
    kwargs: dict,
    latency: float,
    expected_failure: str,
) -> None:
    """기준 하나만 어긋나도 그 항목이 미달로 잡힌다"""
    assessment = evaluate_adoption(
        build_band_summary(**kwargs),
        p95_latency_ms=latency,
    )

    assert assessment.verdict == FAIL
    assert [c.name for c in assessment.failed] == [expected_failure]


def test_boundary_values_count_as_met() -> None:
    """기준값과 정확히 같으면 만족으로 본다"""
    criteria = AdoptionCriteria()
    assessment = evaluate_adoption(
        build_band_summary(
            coverage_recall=criteria.min_coverage_recall,
            alert_false_positive_rate=criteria.max_alert_false_positive_rate,
            uncertain_normal_share=criteria.max_uncertain_normal_share,
            normal_count=criteria.min_normal_samples,
            phishing_count=criteria.min_phishing_samples,
        ),
        p95_latency_ms=criteria.max_p95_latency_ms,
    )

    assert assessment.verdict == PASS


def test_criteria_defaults_are_fixed() -> None:
    """기준값은 #85에서 확정했고 임의로 바뀌면 안 된다"""
    criteria = AdoptionCriteria()

    assert criteria.min_coverage_recall == 0.95
    assert criteria.max_alert_false_positive_rate == 0.01
    assert criteria.max_uncertain_normal_share == 0.133
    assert criteria.max_p95_latency_ms == 50.0
    assert criteria.min_normal_samples == 300
    assert criteria.min_phishing_samples == 60


def test_assessment_is_serialisable() -> None:
    """리포트에 그대로 기록할 수 있어야 한다"""
    assessment = evaluate_adoption(
        build_band_summary(),
        p95_latency_ms=10.0,
    )
    payload = assessment.to_dict()

    assert payload["verdict"] == PASS
    assert {c["name"] for c in payload["criteria"]} == {
        "normal_sample_size",
        "phishing_sample_size",
        "coverage_recall",
        "alert_false_positive_rate",
        "uncertain_normal_share",
        "p95_latency_ms",
    }


def test_v3_measurement_fails_the_gate() -> None:
    """현행 v3 실측치는 기준에 미달한다

    실제 문자 평가셋에서 자동 경고 오탐 5.4%, '의심' 정상 16.2%로,
    상한 1%와 10%를 각각 넘는다. 이 판정이 바뀌면 기준이나 모델 중 하나가
    바뀐 것이므로 문서와 함께 확인해야 한다.
    """
    assessment = evaluate_adoption(
        build_band_summary(
            coverage_recall=0.9518,
            alert_false_positive_rate=0.05405,
            uncertain_normal_share=0.1622,
            normal_count=74,
            phishing_count=83,
        ),
        p95_latency_ms=10.25,
    )

    assert assessment.verdict == FAIL
    assert {c.name for c in assessment.failed} == {
        "alert_false_positive_rate",
        "uncertain_normal_share",
    }
    assert [c.name for c in assessment.unmeasurable] == ["normal_sample_size"]
