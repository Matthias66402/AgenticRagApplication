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
    # Reasoning-Aufwand des Planers (minimal, low, medium, high); leer = Standard des Modells
    llm_reasoning_effort: str = ""
    # Anfragelimit des API-Keys je Modell (Anfragen pro Minute) für alle Agenten; 0 = nicht drosseln
    llm_requests_per_minute: int = 0
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dim: int = 1536
    # Bewertet Suchtreffer der Wissensbasis vor der Weitergabe an den Agenten; leer = nur Mindestähnlichkeit.
    # Anderes Modell als der Recherche-Agent: Eine Recherche löst je Suche eine Bewertung aus; mit gemeinsamem
    # Anfragelimit (je Modell) musste sie sonst oft auf das nächste Minutenfenster warten.
    relevance_model: str = "openai:gpt-4o-mini"
    # Recherche-Agent, an den der Planer die Suche in der Wissensbasis delegiert (app.agent.research)
    research_model: str = "openai:gpt-4.1-mini"
    # Bestimmt vorab, was der Graph vor dem Planer ausführt (app.agent.intake); leer = aus.
    # gpt-4o-mini ordnet Übernachtungsarten zuverlässiger zu als gpt-4.1-mini (z. B. Shelter → bivouac statt hut).
    # Es teilt sich das Anfragelimit mit der Relevanzprüfung, die seit der gebündelten Bewertung nur 1-2 Anfragen je
    # Recherche braucht.
    intake_model: str = "openai:gpt-4o-mini"
    # Längste Tour in Tagen (Obergrenze für AdventurePlan.duration_days). Fernwege wie der Camino del Norte brauchen
    # bei ~25 km/Tag 35-40 Tage; bei 30 bot das Modell an, die Tour in zwei Pläne aufzuteilen.
    max_duration_days: int = 60

    ors_api_key: str = ""
    nominatim_user_agent: str = "AgenticRagAdventurePlanner/0.1"
    overpass_urls: str = "https://overpass-api.de/api/interpreter,https://overpass.private.coffee/api/interpreter"
    http_timeout_seconds: float = 20.0
    # plan_route(avoid_main_roads): Hauptstraßen per Overpass erkennen und umfahren. Aus, solange Overpass oft
    # überlastet ist (Abenteuer 210: Aufrufe ~35 s oder ohne Ergebnis); dann normal geroutet mit Hinweis.
    avoid_main_roads_enabled: bool = False
    # Kartenhintergrund im PDF-Export; leer = nur Route und Marker ohne Karte
    map_tile_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    # Fahrplanauskunft für An- und Abreise (MOTIS-API, z. B. Transitous); leer = aus
    transit_api_url: str = "https://api.transitous.org/api/v1/plan"

    cors_origins: str = "http://localhost:5180"

    # Tracing mit Langfuse (siehe app/observability.py); ohne Keys aus
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_tracing_enabled: bool = True

    @property
    def overpass_url_list(self) -> list[str]:
        return [u.strip() for u in self.overpass_urls.split(",") if u.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
