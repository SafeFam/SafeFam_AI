import math
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.analysis.execution import (
    AnalysisExecution,
    AnalysisExecutionStatus,
)
from app.analysis.schemas import (
    ContributionBreakdown,
    EvidenceCategory,
    EvidenceItem,
    RiskGrade,
    SmishingAnalysisResponse,
)
from app.infrastructure.rabbitmq.result_factory import (
    AnalysisResultEventFactory,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisRequestedEvent,
    TextAnalysisMethod,
)


def _request() -> AnalysisRequestedEvent:
    return AnalysisRequestedEvent.model_validate(
        {
            "schemaVersion": "1.0",
            "eventId": str(uuid4()),
            "analysisId": 123,
            "clientMessageId": "sms-result-001",
            "traceId": str(uuid4()),
            "occurredAt": datetime.now(timezone.utc),
            "payload": {
                "sender": "1588-0000",
                "content": "분석 대상 문자",
                "receivedAt": datetime.now(timezone.utc),
                "source": "AUTO",
            },
        }
    )


def _result(
    *,
    status: str = "SUCCESS",
) -> SmishingAnalysisResponse:
    return SmishingAnalysisResponse(
        status=status,
        message="analysis result",
        final_score=82 if status == "SUCCESS" else 40,
        risk_grade=(RiskGrade.HIGH if status == "SUCCESS" else RiskGrade.MEDIUM),
        contribution_breakdown=ContributionBreakdown(
            llm=42,
            hybrid_url=25,
            rules=15,
        ),
        text_analysis=(
            {
                "result": {
                    "risk_score": 84,
                    "grade": "DANGEROUS",
                    "reason": "금융기관 사칭",
                    "evidence": ["인증 요구"],
                },
                "self_model": {
                    "risk_score": 70,
                    "confidence": 0.72,
                },
                "llm_called": True,
                "llm_available": True,
                "llm_provider": "AWS_BEDROCK",
                "llm_model": "test-model",
                "decision_source": "LLM",
                "routing_reason": "UNCERTAIN_SELF_MODEL_PREDICTION",
                "fallback_applied": False,
            }
            if status == "SUCCESS"
            else None
        ),
        url_analysis=(
            {
                "has_url": True,
                "original_url": "https://short.example/a",
                "origin_url": "https://example.test/login",
                "is_url_malicious": True,
                "url_risk_score": 0.9,
                "engine_source": "Hybrid-Engine",
                "error_message": None,
            }
            if status == "SUCCESS"
            else None
        ),
        rule_analysis=(
            {
                "rule_score": 75,
                "matched_rules": ["financial impersonation"],
                "has_malicious_domain_pattern": True,
                "institution_match": {
                    "checked": True,
                    "mismatch": True,
                    "institution": "국민은행",
                    "official_domains": ["kbstar.com"],
                    "text_domain": "kb-bank-security.xyz",
                },
            }
            if status == "SUCCESS"
            else None
        ),
        evidence=(
            [
                EvidenceItem(
                    category=EvidenceCategory.INSTITUTION_IMPERSONATION,
                    title="기관 사칭",
                    description="국민은행을 언급했지만 공식 도메인이 아닙니다.",
                )
            ]
            if status == "SUCCESS"
            else []
        ),
    )


@pytest.mark.parametrize(
    ("status", "event_type", "failed_tracks"),
    [
        (
            AnalysisExecutionStatus.COMPLETED,
            AnalysisEventType.COMPLETED,
            (),
        ),
        (
            AnalysisExecutionStatus.PARTIAL,
            AnalysisEventType.PARTIAL,
            ("URL:VIRUSTOTAL",),
        ),
    ],
)
def test_factory_maps_successful_execution(
    status: AnalysisExecutionStatus,
    event_type: AnalysisEventType,
    failed_tracks: tuple[str, ...],
) -> None:
    request = _request()
    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=status,
            result=_result(),
            failed_tracks=failed_tracks,
        ),
    )

    assert event.eventType == event_type
    assert event.causationId == request.eventId
    assert event.analysisId == request.analysisId
    assert event.traceId == request.traceId
    assert event.payload.finalScore == 82
    assert event.payload.riskGrade == "HIGH"
    assert event.payload.phishingType == "OTHER"
    assert event.payload.rawScores.text == 84
    assert event.payload.rawScores.url == 90
    assert event.payload.rawScores.rules == 75
    assert event.payload.failedTracks == list(failed_tracks)
    assert event.payload.textAnalysis is not None
    assert event.payload.textAnalysis.method == TextAnalysisMethod.STACKING_LLM
    assert event.payload.textAnalysis.selfModelScore == 70
    assert event.payload.textAnalysis.selfModelConfidence == pytest.approx(0.72)
    assert event.payload.textAnalysis.geminiCalled is True
    assert event.payload.textAnalysis.llmCalled is True
    assert event.payload.textAnalysis.llmProvider == "AWS_BEDROCK"
    assert event.payload.textAnalysis.llmModel == "test-model"
    assert event.payload.textAnalysis.decisionSource == "LLM"
    assert event.payload.textAnalysis.routingReason == (
        "UNCERTAIN_SELF_MODEL_PREDICTION"
    )
    assert event.payload.textAnalysis.fallbackApplied is False
    assert event.payload.ruleAnalysis is not None
    assert event.payload.ruleAnalysis.institutionMatch is not None
    assert event.payload.ruleAnalysis.institutionMatch.checked is True
    assert event.payload.ruleAnalysis.institutionMatch.mismatch is True
    assert event.payload.ruleAnalysis.institutionMatch.institution == "국민은행"
    assert event.payload.ruleAnalysis.institutionMatch.officialDomains == [
        "kbstar.com"
    ]
    assert (
        event.payload.ruleAnalysis.institutionMatch.textDomain
        == "kb-bank-security.xyz"
    )
    assert len(event.payload.evidenceCards) == 1
    assert event.payload.evidenceCards[0].category == "INSTITUTION_IMPERSONATION"
    assert event.payload.evidenceCards[0].title == "기관 사칭"
    assert event.payload.evidenceCards[0].description == (
        "국민은행을 언급했지만 공식 도메인이 아닙니다."
    )


def test_factory_maps_empty_evidence_cards_on_failure() -> None:
    request = _request()
    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.FAILED,
            result=_result(status="ERROR"),
            failed_tracks=("PIPELINE",),
        ),
    )

    assert event.payload.evidenceCards == []
    assert event.payload.phishingType is None


def test_factory_populates_phishing_type_from_request_content() -> None:
    request = _request()
    request.payload.content = "검찰 수사관입니다. 사건 확인이 필요합니다."

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=_result(),
            failed_tracks=(),
        ),
    )

    assert event.payload.phishingType == "GOVERNMENT_AGENCY"
    assert event.model_dump(mode="json")["payload"]["phishingType"] == (
        "GOVERNMENT_AGENCY"
    )


def test_factory_maps_institution_match_not_checked() -> None:
    """기관명이 언급되지 않아 대조 자체를 안 한 경우(checked=False)도 그대로 반영한다."""
    request = _request()
    result = _result()
    result.rule_analysis["institution_match"] = {
        "checked": False,
        "mismatch": False,
        "institution": None,
        "official_domains": [],
        "text_domain": None,
    }

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=result,
            failed_tracks=(),
        ),
    )

    assert event.payload.ruleAnalysis is not None
    assert event.payload.ruleAnalysis.institutionMatch is not None
    assert event.payload.ruleAnalysis.institutionMatch.checked is False
    assert event.payload.ruleAnalysis.institutionMatch.institution is None
    assert event.payload.ruleAnalysis.institutionMatch.mismatch is False
    assert event.payload.ruleAnalysis.institutionMatch.officialDomains == []
    assert event.payload.ruleAnalysis.institutionMatch.textDomain is None


def test_factory_maps_failed_execution_without_message_content() -> None:
    request = _request()
    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.FAILED,
            result=_result(status="ERROR"),
            failed_tracks=("PIPELINE",),
        ),
    )

    assert event.eventType == AnalysisEventType.FAILED
    assert event.payload.finalScore is None
    assert event.payload.riskGrade is None
    assert event.payload.failureCode == "PIPELINE_FAILED"
    assert request.payload.content not in event.model_dump_json()


def test_factory_uses_deterministic_result_event_id() -> None:
    request = _request()
    execution = AnalysisExecution(
        status=AnalysisExecutionStatus.COMPLETED,
        result=_result(),
        failed_tracks=(),
    )
    factory = AnalysisResultEventFactory()

    first = factory.create(
        request=request,
        execution=execution,
    )
    second = factory.create(
        request=request,
        execution=execution,
    )

    assert first.eventId == second.eventId


def test_factory_maps_machine_readable_url_error_codes() -> None:
    request = _request()
    result = _result()
    result.url_analysis["provider_error_codes"] = {
        "VIRUSTOTAL": "RATE_LIMITED",
        "GSB": "TIMEOUT",
    }

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.PARTIAL,
            result=result,
            failed_tracks=(
                "URL:GSB",
                "URL:VIRUSTOTAL",
            ),
        ),
    )

    assert event.payload.urlAnalysis is not None
    assert event.payload.urlAnalysis.errorCode == (
        "GSB:TIMEOUT;VIRUSTOTAL:RATE_LIMITED"
    )


def test_factory_identifies_llm_only_text_analysis() -> None:
    request = _request()
    result = _result()
    result.text_analysis = {
        "engine": "llm",
        "result": {
            "risk_score": 55,
            "grade": "SUSPICIOUS",
            "evidence": [],
        },
    }

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=result,
            failed_tracks=(),
        ),
    )

    assert event.payload.textAnalysis is not None
    assert event.payload.textAnalysis.method == TextAnalysisMethod.LLM


def test_factory_reads_legacy_gemini_called_alias() -> None:
    request = _request()
    result = _result()
    assert result.text_analysis is not None
    result.text_analysis.pop("llm_called")
    result.text_analysis["gemini_called"] = True
    result.text_analysis["llm_available"] = False

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=result,
            failed_tracks=(),
        ),
    )

    assert event.payload.textAnalysis is not None
    assert event.payload.textAnalysis.llmCalled is True
    assert event.payload.textAnalysis.geminiCalled is True
    assert event.payload.textAnalysis.method == TextAnalysisMethod.STACKING_LLM


def test_factory_maps_unit_url_score_to_one_hundred() -> None:
    request = _request()
    result = _result()
    result.url_analysis["url_risk_score"] = 1

    event = AnalysisResultEventFactory().create(
        request=request,
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=result,
            failed_tracks=(),
        ),
    )

    assert event.payload.rawScores.url == 100
    assert event.payload.urlAnalysis is not None
    assert event.payload.urlAnalysis.score == 100


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [
        (-0.5, 0.0),
        (1.5, 1.0),
        (float("nan"), None),
        (float("inf"), None),
        (True, None),
        ("0.8", None),
    ],
)
def test_factory_normalizes_self_model_confidence(
    raw_confidence,
    expected,
) -> None:
    result = _result()
    result.text_analysis["self_model"]["confidence"] = (
        raw_confidence
    )

    event = AnalysisResultEventFactory().create(
        request=_request(),
        execution=AnalysisExecution(
            status=AnalysisExecutionStatus.COMPLETED,
            result=result,
            failed_tracks=(),
        ),
    )

    confidence = event.payload.textAnalysis.selfModelConfidence
    if expected is None:
        assert confidence is None
    else:
        assert math.isclose(confidence, expected)
