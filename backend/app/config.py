import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/krishiai")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    DATA_GOV_API_KEY: str = os.getenv("DATA_GOV_API_KEY", "") or os.getenv("DATA_GOV_IN_API_KEY", "")
    
    class Config:
        env_file = ".env"

settings = Settings()
