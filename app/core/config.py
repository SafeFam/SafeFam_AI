from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    VIRUSTOTAL_API_KEY: str | None = None
    GOOGLE_SAFE_BROWSING_API_KEY: str | None = None


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


    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

settings = Settings()
