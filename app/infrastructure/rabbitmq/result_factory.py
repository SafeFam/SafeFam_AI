import math
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from app.analysis.execution import (
    AnalysisExecution,
    AnalysisExecutionStatus,
)
from app.infrastructure.rabbitmq.schemas import (
    AnalysisEventType,
    AnalysisRequestedEvent,
    AnalysisResultEvent,
    AnalysisResultPayload,
    EvidenceCardDetail,
    InstitutionMatchDetail,
    RawScores,
    RuleAnalysisDetail,
    TextAnalysisDetail,
    TextAnalysisMethod,
    UrlAnalysisDetail,
    WeightedContributions,
)


class AnalysisResultEventFactory:
    """내부 분석 실행 결과를 메시징 규격 이벤트로 변환"""

    def create(
        self,
        request: AnalysisRequestedEvent,
        execution: AnalysisExecution,
    ) -> AnalysisResultEvent:
        """요청 이벤트 데이터와 실행 결과를 결합하여 전달할 결과 이벤트를 생성"""

        # 내부 분석 상태(COMPLETED, PARTIAL, FAILED)를 외부 메시징 이벤트 타입으로 맵핑
        event_type = {
            AnalysisExecutionStatus.COMPLETED: AnalysisEventType.COMPLETED,
            AnalysisExecutionStatus.PARTIAL: AnalysisEventType.PARTIAL,
            AnalysisExecutionStatus.FAILED: AnalysisEventType.FAILED,
        }[execution.status]

        result = execution.result
        result_event_id = uuid5(
            NAMESPACE_URL,
            (f"safefam:analysis-result:{request.eventId}:{event_type.value}"),
        )

        # 원본 요청의 식별자를 포함하여 결과 이벤트 구성
        return AnalysisResultEvent(
            schemaVersion="1.0",
            eventId=result_event_id,
            causationId=request.eventId,
            analysisId=request.analysisId,
            clientMessageId=request.clientMessageId,
            traceId=request.traceId,
            occurredAt=datetime.now(timezone.utc),
            eventType=event_type,
            payload=build_payload(
                result=result,
                failed_tracks=execution.failed_tracks,
            ),
        )


def build_payload(
    result,
    failed_tracks: tuple[str, ...],
) -> AnalysisResultPayload:
    """내부 분석 응답을 외부 결과 이벤트 payload로 변환합니다."""
    text = result.text_analysis or {}
    text_result = text.get("result") or {}
    url = result.url_analysis or {}
    rules = result.rule_analysis or {}

    text_score = _integer_score(text_result.get("risk_score"))
    url_score = _url_score(url.get("url_risk_score"))
    rule_score = _integer_score(rules.get("rule_score"))

    is_failed = result.status == "ERROR"

    return AnalysisResultPayload(
        finalScore=(None if is_failed else result.final_score),
        riskGrade=(
            None
            if is_failed
            else getattr(
                result.risk_grade,
                "value",
                result.risk_grade,
            )
        ),
        phishingType=None,
        rawScores=RawScores(
            text=text_score,
            url=url_score,
            rules=rule_score,
        ),
        weightedContributions=(
            None
            if is_failed
            else WeightedContributions(
                text=result.contribution_breakdown.llm,
                url=result.contribution_breakdown.hybrid_url,
                rules=result.contribution_breakdown.rules,
            )
        ),
        textAnalysis=(
            None if is_failed or not text else _text_detail(text, failed_tracks)
        ),
        urlAnalysis=(None if is_failed or not url else _url_detail(url)),
        ruleAnalysis=(None if is_failed or not rules else _rule_detail(rules)),
        evidenceCards=_evidence_cards(result.evidence),
        failedTracks=list(failed_tracks),
        failureCode=("PIPELINE_FAILED" if is_failed else None),
    )


def _evidence_cards(evidence) -> list[EvidenceCardDetail]:
    return [
        EvidenceCardDetail(
            category=item.category.value,
            title=item.title,
            description=item.description,
        )
        for item in (evidence or [])
    ]


def _text_detail(
    text: dict,
    failed_tracks: tuple[str, ...],
) -> TextAnalysisDetail:
    result = text.get("result") or {}
    self_model = text.get("self_model") or {}

    decision_source = text.get(
        "decision_source"
    )
    llm_called = bool(
        text.get(
            "llm_called",
            text.get(
                "gemini_called",
                False,
            ),
        )
    )

    if result.get("grade") == "UNKNOWN":
        method = TextAnalysisMethod.UNAVAILABLE

    elif decision_source == "STACKING":
        method = TextAnalysisMethod.STACKING

    elif decision_source == "STACKING_FALLBACK":
        method = (
            TextAnalysisMethod.STACKING_FALLBACK
        )

    elif llm_called:
        method = (
            TextAnalysisMethod.STACKING_LLM
        )

    else:
        method = TextAnalysisMethod.LLM

    return TextAnalysisDetail(
        method=method,
        score=_integer_score(
            result.get("risk_score")
        ),
        grade=result.get("grade"),
        reason=result.get("reason"),
        evidence=result.get("evidence") or [],
        failedEngines=[
            track.removeprefix("TEXT:")
            for track in failed_tracks
            if track.startswith("TEXT:")
        ],
        selfModelScore=_integer_score(
            self_model.get("risk_score")
        ),
        selfModelConfidence=(
            _confidence(
                self_model.get("confidence")
            )
        ),
        llmCalled=llm_called,
        llmProvider=text.get("llm_provider"),
        llmModel=text.get("llm_model"),
        llmUsage=text.get("llm_usage") or {},
        llmLatencyMs=text.get("llm_latency_ms"),
        llmFromCache=text.get("llm_from_cache") is True,

        # Deprecated: temporary SafeFam_BE compatibility.
        geminiCalled=llm_called,
        decisionSource=decision_source,
        routingReason=text.get(
            "routing_reason"
        ),
        fallbackApplied=bool(
            text.get("fallback_applied")
        ),
    )


def _url_detail(url: dict) -> UrlAnalysisDetail:
    return UrlAnalysisDetail(
        hasUrl=bool(url.get("has_url")),
        originalUrl=url.get("original_url"),
        tracedUrl=url.get("origin_url"),
        malicious=url.get("is_url_malicious"),
        score=_url_score(url.get("url_risk_score")),
        engineSource=url.get("engine_source"),
        errorCode=_url_error_code(url),
    )


def _rule_detail(rules: dict) -> RuleAnalysisDetail:
    return RuleAnalysisDetail(
        score=_integer_score(rules.get("rule_score")) or 0,
        matchedRules=rules.get("matched_rules") or [],
        maliciousDomainPattern=bool(
            rules.get(
                "has_malicious_domain_pattern",
                False,
            )
        ),
        institutionMatch=_institution_match_detail(
            rules.get("institution_match")
        ),
    )


def _institution_match_detail(
    institution_match: dict | None,
) -> InstitutionMatchDetail | None:
    if not institution_match:
        return None

    return InstitutionMatchDetail(
        checked=bool(institution_match.get("checked", False)),
        mismatch=bool(institution_match.get("mismatch", False)),
        institution=institution_match.get("institution"),
        officialDomains=list(
            institution_match.get("official_domains") or []
        ),
        textDomain=institution_match.get("text_domain"),
        phoneChecked=bool(institution_match.get("phone_checked", False)),
        phoneMismatch=bool(institution_match.get("phone_mismatch", False)),
        textPhoneNumbers=list(
            institution_match.get("text_phone_numbers") or []
        ),
        officialPhoneNumbers=list(
            institution_match.get("official_phone_numbers") or []
        ),
    )


def _integer_score(value) -> int | None:
    if value is None:
        return None
    return max(0, min(100, round(float(value))))


def _confidence(value) -> float | None:
    """Confidence를 외부 계약의 0.0~1.0 범위로 정규화."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        return None

    numeric = float(value)
    if not math.isfinite(numeric):
        return None

    return min(max(numeric, 0.0), 1.0)


def _url_error_code(url: dict) -> str | None:
    provider_codes = url.get("provider_error_codes") or {}

    if not provider_codes:
        return None

    return ";".join(
        f"{provider}:{code}" for provider, code in sorted(provider_codes.items())
    )


def _url_score(value) -> int | None:
    if value is None:
        return None

    numeric = float(value)
    if 0 <= numeric <= 1:
        numeric *= 100
    return _integer_score(numeric)
