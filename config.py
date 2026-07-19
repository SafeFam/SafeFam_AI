import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    VIRUSTOTAL_API_KEY: str = os.getenv("VIRUSTOTAL_API_KEY", "")
    GOOGLE_SAFE_BROWSING_API_KEY: str = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")

    def __init__(self):
        if not self.VIRUSTOTAL_API_KEY:
            print("VIRUSTOTAL_API_KEY가 설정되지 않았습니다.")
        if not self.GOOGLE_SAFE_BROWSING_API_KEY:
            print("GOOGLE_SAFE_BROWSING_API_KEY가 설정되지 않았습니다.")

settings = Settings()