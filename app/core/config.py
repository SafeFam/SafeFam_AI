from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # 기본 애플리케이션 설정
    ENV: Literal["local", "test", "prod"] = "local"
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # 외부 보안 API
    VIRUSTOTAL_API_KEY: str | None = None
    GOOGLE_SAFE_BROWSING_API_KEY: str | None = None
    MOCK_SECURITY_API: bool = False
    
    EXTERNAL_API_MAX_RETRIES: int = Field(
        default=1,
        ge=0,
        le=3,
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

    # LLM / Amazon Bedrock
    LLM_PROVIDER: Literal["bedrock"] = "bedrock"
    AWS_REGION: str = "us-east-1"
    AWS_PROFILE: str | None = None

    BEDROCK_MODEL_ID: str = (
        "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    LLM_TIMEOUT_SECONDS: float = Field(
        default=15.0,
        gt=0,
        le=120,
    )

    LLM_MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    LLM_MAX_OUTPUT_TOKENS: int = Field(
        default=1_024,
        ge=128,
        le=8_192,
    )

    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    # 입력 크기 제한 (DoS·LLM 비용 방지, issue #120)
    # SafeFam_BE가 계약상 게이트키퍼이므로 BE의 @Size 검증값과 정합을 맞춘다.
    # - 분석 content: BE AnalysisRequest @Size(max = 5000)
    # - 챗 content:  BE ChatMessage @Size(max = 2000)
    MAX_ANALYSIS_CONTENT_LENGTH: int = Field(
        default=5_000,
        ge=1,
    )
    MAX_CHAT_CONTENT_LENGTH: int = Field(
        default=2_000,
        ge=1,
    )
    MAX_CHAT_MESSAGES: int = Field(
        default=40,
        ge=1,
    )

    # Stacking 모델 임계값
    STACKING_NORMAL_PROBABILITY_MAX: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )

    STACKING_PHISHING_PROBABILITY_MIN: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
    )

    # 모델 파일 경로
    NAIVE_BAYES_MODEL_PATH: Path = Path(
        "data_science/SMSModel/artifacts/phishing_model_artifact.pkl"
    )

    NAIVE_BAYES_VECTORIZER_PATH: Path = Path(
        "data_science/SMSModel/artifacts/phishing_vectorizer.pkl"
    )

    # RabbitMQ 토폴로지
    RABBITMQ_URL: str = "amqp://safefam:safefam-local@localhost:5672/"
    RABBITMQ_ANALYSIS_EXCHANGE: str = "safefam.analysis"
    RABBITMQ_ANALYSIS_REQUEST_QUEUE: str = "safefam.analysis.requested.q"
    RABBITMQ_ANALYSIS_REQUEST_ROUTING_KEY: str = "analysis.requested.v1"
    RABBITMQ_ANALYSIS_COMPLETED_ROUTING_KEY: str = "analysis.completed.v1"
    RABBITMQ_ANALYSIS_PARTIAL_ROUTING_KEY: str = "analysis.partial.v1"
    RABBITMQ_ANALYSIS_FAILED_ROUTING_KEY: str = "analysis.failed.v1"
    RABBITMQ_ANALYSIS_DLQ: str = "safefam.analysis.requested.dlq"
    RABBITMQ_ANALYSIS_DLQ_ROUTING_KEY: str = "analysis.requested.dead.v1"

    # RabbitMQ 실행 설정
    RABBITMQ_CONSUMER_ENABLED: bool = True
    RABBITMQ_PREFETCH_COUNT: int = Field(default=1, ge=1)

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

    # Validator
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
            "AWS_REGION": self.AWS_REGION,
            "BEDROCK_MODEL_ID": self.BEDROCK_MODEL_ID,
            "VIRUSTOTAL_API_KEY": self.VIRUSTOTAL_API_KEY,
            "GOOGLE_SAFE_BROWSING_API_KEY": (
                self.GOOGLE_SAFE_BROWSING_API_KEY
            ),
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
