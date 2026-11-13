import os
from pathlib import Path
from  dotenv import load_dotenv
load_dotenv()
class Settings:

    PROJECT_NAME : str = "PTBN"

    SECRET_KEY : str = os.getenv("SECRET_KEY" , "temporary_low_security_key")
    ALGORITHM : str = os.getenv("ALGORITHM" , "HS256")

    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "lax")

    DATABASE_URL : str = os.getenv("DATABASE_URL")

    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD")
    MAIL_FROM: str = os.getenv("MAIL_FROM")
    MAIL_PORT: int = int(os.getenv("MAIL_PORT", "587"))
    MAIL_SERVER: str = os.getenv("MAIL_SERVER")

    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    OHLCV : str = os.getenv("OHLCV")


settings = Settings()