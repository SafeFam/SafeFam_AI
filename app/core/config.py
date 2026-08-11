from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENV: Literal["local", "test", "prod"] = "local"
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-flash-latest"
    VIRUSTOTAL_API_KEY: str | None = None
    GOOGLE_SAFE_BROWSING_API_KEY: str | None = None
    MOCK_SECURITY_API: bool = False

    # Stacking 자체 모델이 확실한 정상이라고 판단하는 최대 확률
    STACKING_NORMAL_PROBABILITY_MAX: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    # Stacking 자체 모델이 확실한 피싱이라고 판단하는 최소 확률
    STACKING_PHISHING_PROBABILITY_MIN: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
    )

    NAIVE_BAYES_MODEL_PATH: Path = Path(
        "data_science/SMSModel/artifacts/phishing_model_artifact.pkl"
    )
    NAIVE_BAYES_VECTORIZER_PATH: Path = Path(
        "data_science/SMSModel/artifacts/phishing_vectorizer.pkl"
    )

    GEMINI_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        gt=0,
    )
    GSB_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )
    VIRUSTOTAL_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )
    URL_TRACE_TIMEOUT_SECONDS: float = Field(
        default=3.0,
        gt=0,
    )
    EXTERNAL_API_MAX_RETRIES: int = Field(
        default=1,
        ge=0,
        le=3,
    )

    RABBITMQ_URL: str = "amqp://safefam:safefam-local@localhost:5672/"
    RABBITMQ_ANALYSIS_EXCHANGE: str = "safefam.analysis"
    RABBITMQ_ANALYSIS_REQUEST_QUEUE: str = "safefam.analysis.requested.q"
    RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY: str = "analysis.requested.v1"
    RABBITMQ_PREFETCH_COUNT: int = Field(default=1, ge=1)
    RABBITMQ_CONSUMER_ENABLED: bool = True

    RABBITMQ_ANALYSIS_COMPLETED_ROUTING_KEY: str = "analysis.completed.v1"
    RABBITMQ_ANALYSIS_PARTIAL_ROUTING_KEY: str = "analysis.partial.v1"
    RABBITMQ_ANALYSIS_FAILED_ROUTING_KEY: str = "analysis.failed.v1"

    RABBITMQ_ANALYSIS_DLQ: str = "safefam.analysis.requested.dlq"
    RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY: str = "analysis.requested.dead.v1"

    RABBITMQ_PUBLISH_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )
    RABBITMQ_SHUTDOWN_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
    )
    RABBITMQ_REQUEUE_BACKOFF_SECONDS: float = Field(
        default=1.0,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_production_settings(self):
        if (
            self.STACKING_NORMAL_PROBABILITY_MAX
            >= self.STACKING_PHISHING_PROBABILITY_MIN
        ):
            raise ValueError(
                "STACKING_NORMAL_PROBABILITY_MAX must be smaller than "
                "STACKING_PHISHING_PROBABILITY_MIN"
            )

        if self.ENV != "prod":
            return self

        required_values = {
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "VIRUSTOTAL_API_KEY": self.VIRUSTOTAL_API_KEY,
            "GOOGLE_SAFE_BROWSING_API_KEY": (self.GOOGLE_SAFE_BROWSING_API_KEY),
            "RABBITMQ_URL": self.RABBITMQ_URL,
        }
        missing = [
            name
            for name, value in required_values.items()
            if value is None or not str(value).strip()
        ]

        local_rabbitmq_url = "amqp://safefam:safefam-local@localhost:5672/"
        if self.RABBITMQ_URL == local_rabbitmq_url:
            missing.append("RABBITMQ_URL")

        if missing:
            raise ValueError(
                "Missing required production settings: "
                + ", ".join(sorted(set(missing)))
            )

        if self.MOCK_SECURITY_API:
            raise ValueError("MOCK_SECURITY_API must be false in production")

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )


settings = Settings()
