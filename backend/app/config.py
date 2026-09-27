import os
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/krishiai"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    DATA_GOV_API_KEY: str = os.getenv("DATA_GOV_API_KEY", "") or os.getenv("DATA_GOV_IN_API_KEY", "")
    
    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def normalize_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgres://") and not v.startswith("postgresql+asyncpg://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        
        # asyncpg requires ssl=require instead of sslmode=require and doesn't accept channel_binding
        if "sslmode=require" in v:
            v = v.replace("sslmode=require", "ssl=require")
        if "&channel_binding=require" in v:
            v = v.replace("&channel_binding=require", "")
        if "?channel_binding=require" in v:
            v = v.replace("?channel_binding=require", "")
        return v
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
