from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.main import validate_model_files


def _production_values() -> dict[str, object]:
    return {
        "ENV": "prod",
        "GEMINI_API_KEY": "test-gemini-key",
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


def test_model_validation_rejects_missing_files(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing-model.pkl"
    missing_vectorizer = tmp_path / "missing-vectorizer.pkl"

    monkeypatch.setattr(
        "app.main.settings.NAIVE_BAYES_MODEL_PATH",
        missing_model,
    )
    monkeypatch.setattr(
        "app.main.settings.NAIVE_BAYES_VECTORIZER_PATH",
        missing_vectorizer,
    )

    with pytest.raises(RuntimeError, match="Required AI model files"):
        validate_model_files()


def test_model_validation_accepts_readable_files(monkeypatch, tmp_path):
    model_path = Path(tmp_path / "model.pkl")
    vectorizer_path = Path(tmp_path / "vectorizer.pkl")
    model_path.write_bytes(b"model")
    vectorizer_path.write_bytes(b"vectorizer")

    monkeypatch.setattr(
        "app.main.settings.NAIVE_BAYES_MODEL_PATH",
        model_path,
    )
    monkeypatch.setattr(
        "app.main.settings.NAIVE_BAYES_VECTORIZER_PATH",
        vectorizer_path,
    )

    validate_model_files()
