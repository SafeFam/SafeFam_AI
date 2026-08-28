import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.main import validate_model_files


def _production_values() -> dict[str, object]:
    return {
        "ENV": "prod",
        "VIRUSTOTAL_API_KEY": "test-virustotal-key",
        "GOOGLE_SAFE_BROWSING_API_KEY": "test-gsb-key",
        "RABBITMQ_URL": "amqp://user:password@rabbitmq:5672/",
        "MOCK_SECURITY_API": False,
    }


def test_production_settings_require_secrets():
    with pytest.raises(ValidationError) as exception_info:
        Settings(ENV="prod", _env_file=None)

    error_message = str(exception_info.value)
    assert "Missing required production settings" in error_message
    assert "input_value" not in error_message


def test_production_settings_reject_mock_mode():
    values = _production_values()
    values["MOCK_SECURITY_API"] = True

    with pytest.raises(ValidationError, match="must be false"):
        Settings(**values, _env_file=None)


def test_production_settings_accept_complete_configuration():
    configured = Settings(**_production_values(), _env_file=None)

    assert configured.ENV == "prod"
    assert configured.MOCK_SECURITY_API is False


def test_external_api_timeout_defaults():
    configured = Settings(_env_file=None)

    assert configured.GSB_TIMEOUT_SECONDS == 5.0
    assert configured.VIRUSTOTAL_TIMEOUT_SECONDS == 5.0
    assert configured.URL_TRACE_TIMEOUT_SECONDS == 3.0


def test_input_size_limit_defaults_match_backend_contract():
    """입력 크기 상한은 SafeFam_BE의 @Size 검증값과 정합을 맞춘다(issue #120).

    BE가 게이트키퍼이므로 값이 어긋나면 한쪽만 통과하는 불일치가 생긴다.
    - 분석 content: BE AnalysisRequest @Size(max = 5000)
    - 챗 content:  BE ChatMessage @Size(max = 2000)
    """
    configured = Settings(_env_file=None)

    assert configured.MAX_ANALYSIS_CONTENT_LENGTH == 5000
    assert configured.MAX_CHAT_CONTENT_LENGTH == 2000
    assert configured.MAX_CHAT_MESSAGES == 40


def test_chat_context_size_limit_defaults():
    """analysisContext(LLM 프롬프트 주입)의 안전 상한 기본값(issue #120)."""
    configured = Settings(_env_file=None)

    assert configured.MAX_CHAT_CONTEXT_TEXT_LENGTH == 2000
    assert configured.MAX_CHAT_INDICATORS == 20


def test_request_body_size_limit_default():
    """HTTP 바디 크기 상한 기본값 1MiB(issue #120)."""
    configured = Settings(_env_file=None)

    assert configured.MAX_REQUEST_BODY_BYTES == 1_048_576


@pytest.mark.parametrize(
    "field_name",
    [
        "MAX_ANALYSIS_CONTENT_LENGTH",
        "MAX_CHAT_CONTENT_LENGTH",
        "MAX_CHAT_MESSAGES",
        "MAX_CHAT_CONTEXT_TEXT_LENGTH",
        "MAX_CHAT_INDICATORS",
        "MAX_REQUEST_BODY_BYTES",
    ],
)
def test_input_size_limits_reject_non_positive(field_name: str):
    with pytest.raises(ValidationError):
        Settings(**{field_name: 0}, _env_file=None)


def test_settings_accept_valid_stacking_probability_bounds():
    configured = Settings(
        STACKING_NORMAL_PROBABILITY_MAX=0.2,
        STACKING_PHISHING_PROBABILITY_MIN=0.8,
        _env_file=None,
    )

    assert configured.STACKING_NORMAL_PROBABILITY_MAX == 0.2
    assert configured.STACKING_PHISHING_PROBABILITY_MIN == 0.8


@pytest.mark.parametrize(
    ("normal_max", "phishing_min"),
    [
        (0.5, 0.5),
        (0.8, 0.2),
    ],
)
def test_settings_reject_invalid_stacking_probability_bounds(
    normal_max: float,
    phishing_min: float,
) -> None:
    with pytest.raises(
        ValidationError,
        match="must be smaller",
    ):
        Settings(
            STACKING_NORMAL_PROBABILITY_MAX=normal_max,
            STACKING_PHISHING_PROBABILITY_MIN=phishing_min,
            _env_file=None,
        )


def test_model_validation_rejects_when_stacking_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.is_stacking_model_loaded", lambda: False)

    with pytest.raises(RuntimeError, match="Stacking model"):
        validate_model_files()


def test_model_validation_accepts_when_stacking_model_available(monkeypatch):
    monkeypatch.setattr("app.main.is_stacking_model_loaded", lambda: True)

    validate_model_files()
