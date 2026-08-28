import json
from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.infrastructure.rabbitmq.schemas import (
    AnalysisRequestedEvent,
    AnalysisSource,
)


@pytest.fixture
def valid_event_data() -> dict:
    return {
        "schemaVersion": "1.0",
        "eventId": "1fb898fa-d89d-4d0b-a43f-a8b00daeb765",
        "analysisId": 123,
        "clientMessageId": "sms-20260728-001",
        "traceId": "2d59c74e-0691-4f01-bde3-c657ba4c90cd",
        "occurredAt": "2026-07-28T01:30:00Z",
        "payload": {
            "sender": "1588-0000",
            "content": "[국민은행] 계좌가 정지되었습니다.",
            "receivedAt": "2026-07-28T10:29:00+09:00",
            "source": "AUTO",
        },
    }


def test_parses_valid_analysis_requested_event(
    valid_event_data: dict,
):
    event = AnalysisRequestedEvent.model_validate(valid_event_data)

    assert event.schemaVersion == "1.0"
    assert event.eventId == UUID("1fb898fa-d89d-4d0b-a43f-a8b00daeb765")
    assert event.analysisId == 123
    assert event.clientMessageId == "sms-20260728-001"
    assert event.traceId == UUID("2d59c74e-0691-4f01-bde3-c657ba4c90cd")
    assert event.payload.source == AnalysisSource.AUTO
    assert event.payload.content == ("[국민은행] 계좌가 정지되었습니다.")


def test_parses_auto_analysis_source(
    valid_event_data: dict,
):
    valid_event_data["payload"]["source"] = "AUTO"

    event = AnalysisRequestedEvent.model_validate(valid_event_data)

    assert event.payload.source == AnalysisSource.AUTO


def test_parses_manual_analysis_source(
    valid_event_data: dict,
):
    valid_event_data["payload"]["source"] = "MANUAL"

    event = AnalysisRequestedEvent.model_validate(valid_event_data)

    assert event.payload.source == AnalysisSource.MANUAL


def test_parses_aware_event_timestamps(
    valid_event_data: dict,
):
    event = AnalysisRequestedEvent.model_validate(valid_event_data)

    assert event.occurredAt.utcoffset() == timedelta(0)
    assert event.payload.receivedAt.utcoffset() == timedelta(hours=9)


@pytest.mark.parametrize("content", ["", " ", "   ", "\t", "\n"])
def test_rejects_blank_content(
    valid_event_data: dict,
    content: str,
):
    valid_event_data["payload"]["content"] = content

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_accepts_content_at_max_length(
    valid_event_data: dict,
):
    valid_event_data["payload"]["content"] = "가" * settings.MAX_ANALYSIS_CONTENT_LENGTH

    event = AnalysisRequestedEvent.model_validate(valid_event_data)

    assert len(event.payload.content) == settings.MAX_ANALYSIS_CONTENT_LENGTH


def test_rejects_content_over_max_length(
    valid_event_data: dict,
):
    valid_event_data["payload"]["content"] = "가" * (
        settings.MAX_ANALYSIS_CONTENT_LENGTH + 1
    )

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_oversized_content_is_not_leaked_in_validation_error(
    valid_event_data: dict,
):
    """hide_input_in_errors=True 로 검증 실패 시 원문(PII)이 에러에 담기지 않아야 한다."""
    secret_marker = "01012345678-비밀번호-보이스피싱"
    valid_event_data["payload"]["content"] = (
        secret_marker + "가" * settings.MAX_ANALYSIS_CONTENT_LENGTH
    )

    with pytest.raises(ValidationError) as exception_info:
        AnalysisRequestedEvent.model_validate(valid_event_data)

    assert secret_marker not in str(exception_info.value)


def test_rejects_unsupported_schema_version(
    valid_event_data: dict,
):
    valid_event_data["schemaVersion"] = "2.0"

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


@pytest.mark.parametrize(
    "field_name",
    [
        "eventId",
        "analysisId",
        "traceId",
        "occurredAt",
        "payload",
    ],
)
def test_rejects_missing_required_event_fields(
    valid_event_data: dict,
    field_name: str,
):
    del valid_event_data[field_name]

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


@pytest.mark.parametrize(
    "field_name",
    [
        "content",
        "receivedAt",
        "source",
    ],
)
def test_rejects_missing_required_payload_fields(
    valid_event_data: dict,
    field_name: str,
):
    del valid_event_data["payload"][field_name]

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


@pytest.mark.parametrize(
    "field_name",
    ["eventId", "traceId"],
)
def test_rejects_invalid_uuid(
    valid_event_data: dict,
    field_name: str,
):
    valid_event_data[field_name] = "not-a-uuid"

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


@pytest.mark.parametrize("analysis_id", [0, -1])
def test_rejects_non_positive_analysis_id(
    valid_event_data: dict,
    analysis_id: int,
):
    valid_event_data["analysisId"] = analysis_id

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


@pytest.mark.parametrize(
    ("field_path", "naive_datetime"),
    [
        (("occurredAt",), "2026-07-28T01:30:00"),
        (
            ("payload", "receivedAt"),
            "2026-07-28T10:29:00",
        ),
    ],
)
def test_rejects_datetime_without_timezone(
    valid_event_data: dict,
    field_path: tuple[str, ...],
    naive_datetime: str,
):
    target = valid_event_data

    for key in field_path[:-1]:
        target = target[key]

    target[field_path[-1]] = naive_datetime

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_rejects_unknown_analysis_source(
    valid_event_data: dict,
):
    valid_event_data["payload"]["source"] = "SMS"

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_rejects_unknown_event_field(
    valid_event_data: dict,
):
    valid_event_data["unknownField"] = "unexpected"

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_rejects_unknown_payload_field(
    valid_event_data: dict,
):
    valid_event_data["payload"]["unknownField"] = "unexpected"

    with pytest.raises(ValidationError):
        AnalysisRequestedEvent.model_validate(valid_event_data)


def test_parses_event_from_json_message(
    valid_event_data: dict,
):
    """JSON 문자열 파싱 테스트."""
    message_body = json.dumps(
        valid_event_data,
        ensure_ascii=False,
    ).encode("utf-8")

    event = AnalysisRequestedEvent.model_validate_json(message_body)

    assert event.analysisId == 123
    assert event.payload.content == ("[국민은행] 계좌가 정지되었습니다.")
