from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeFam-AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    VIRUSTOTAL_API_KEY: str = Field(..., env="VIRUSTOTAL_API_KEY")
    GOOGLE_SAFE_BROWSING_API_KEY: str = Field(..., env="GOOGLE_SAFE_BROWSING_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" 
    )

settings = Settings()