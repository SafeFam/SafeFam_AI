from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    VIRUSTOTAL_API_KEY: str = "default_key_here"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()