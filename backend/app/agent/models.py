"""Modell und Modelleinstellungen für die Agenten (Planer, Recherche-Agent, Relevanzprüfung)."""

from pydantic_ai.models import Model
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.settings import ModelSettings

from app.agent.rate_limit import RateLimitedModel, shared_budget

# Ausgaben im strict-Modus: Ohne Constrained Decoding hängt gpt-5-mini gelegentlich zehntausende Leerzeichen an
# die JSON-Argumente an und braucht dafür Minuten. Das Tokenlimit fängt solche Ausreißer zusätzlich ab
# (ein normaler Plan inkl. Reasoning braucht 2.000-5.000 Tokens).
MAX_OUTPUT_TOKENS = 12_000


def model_settings(model: str, reasoning_effort: str = "") -> ModelSettings:
    settings: ModelSettings = {"max_tokens": MAX_OUTPUT_TOKENS}
    provider, _, name = model.partition(":")
    # Ohne angeforderte Zusammenfassung liefern OpenAI-Reasoning-Modelle leere Thinking-Teile; mit ihr ist im Tracing
    # nachvollziehbar, warum der Agent ein Tool aufruft. Nicht-Reasoning-Modelle lehnen beide Parameter ab.
    if provider.startswith("openai") and openai_model_profile(name).get("openai_supports_reasoning"):
        settings["openai_reasoning_summary"] = "detailed"  # type: ignore[typeddict-unknown-key]
        if reasoning_effort:
            settings["openai_reasoning_effort"] = reasoning_effort  # type: ignore[typeddict-unknown-key]
    return settings


def build_model(model: str, requests_per_minute: int = 0) -> Model | str:
    """Mit Limit gedrosselt (siehe app.agent.rate_limit); ohne Limit als Name, den PydanticAI erst bei Bedarf
    auflöst (z. B. in Tests ohne API-Key). Agenten mit demselben Modell teilen sich das Limit."""
    if requests_per_minute <= 0:
        return model
    return RateLimitedModel(model, requests_per_minute, budget=shared_budget(model, requests_per_minute))
