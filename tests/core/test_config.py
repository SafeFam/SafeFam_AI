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
