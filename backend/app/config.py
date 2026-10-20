import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

# Provider-SDKs (OpenAI, Anthropic, ...) lesen ihre API-Keys aus os.environ.
load_dotenv(ENV_FILE)
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    database_url: str = "postgresql+asyncpg://adventure:adventure@localhost:5440/adventure"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 12 * 60

    admin_email: str = "admin@example.com"
    admin_password: str = "admin12345"

    llm_model: str = "openai:gpt-5-mini"
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dim: int = 1536

    ors_api_key: str = ""
    nominatim_user_agent: str = "AgenticRagAdventurePlanner/0.1"
    overpass_urls: str = "https://overpass-api.de/api/interpreter,https://overpass.private.coffee/api/interpreter"
    http_timeout_seconds: float = 20.0

    cors_origins: str = "http://localhost:5180"

    @property
    def overpass_url_list(self) -> list[str]:
        return [u.strip() for u in self.overpass_urls.split(",") if u.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
