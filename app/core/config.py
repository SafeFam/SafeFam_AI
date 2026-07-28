from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    VIRUSTOTAL_API_KEY: str | None = None
    GOOGLE_SAFE_BROWSING_API_KEY: str | None = None

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


    RABBITMQ_URL: str = (
        "amqp://safefam:safefam-local@localhost:5672/"
    )
    RABBITMQ_ANALYSIS_EXCHANGE: str = "safefam.analysis"
    RABBITMQ_ANALYSIS_REQUEST_QUEUE: str = (
        "safefam.analysis.requested.q"
    )
    RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY: str = (
        "analysis.requested.v1"
    )
    RABBITMQ_PREFETCH_COUNT: int = Field(default=1, ge=1)
    RABBITMQ_CONSUMER_ENABLED: bool = True

    RABBITMQ_ANALYSIS_COMPLETED_ROUTING_KEY: str = (
    "analysis.completed.v1"
    )
    RABBITMQ_ANALYSIS_PARTIAL_ROUTING_KEY: str = (
        "analysis.partial.v1"
    )
    RABBITMQ_ANALYSIS_FAILED_ROUTING_KEY: str = (
        "analysis.failed.v1"
    )

    RABBITMQ_ANALYSIS_DLQ: str = (
        "safefam.analysis.requested.dlq"
    )
    RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY: str = (
        "analysis.requested.dead.v1"
    )

    RABBITMQ_PUBLISH_TIMEOUT_SECONDS: float = Field(
        default=5.0,
        gt=0,
    )
    RABBITMQ_SHUTDOWN_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

settings = Settings()
