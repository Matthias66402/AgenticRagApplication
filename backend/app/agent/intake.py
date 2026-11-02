"""Intake: Ein kleines LLM bestimmt vorab die Orte, die der Planer für die aktuelle Nachricht nachschlagen muss.

Der Graph (app.agent.graph) sucht diese Orte per `geocode`, bevor der Planer startet, und fügt Aufrufe und Ergebnisse
in dessen Verlauf ein. Das spart dem Planer die erste Runde, in der er sonst nur Start und Ziel nachschlägt. Ein
Fehler hier ist unkritisch: Der Planer kann jeden Ort selbst (erneut) suchen.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.models import build_model
from app.config import get_settings

MAX_OUTPUT_TOKENS = 1000
# Intake darf die Planung nicht ausbremsen: ohne Wiederholungen nach kurzer Zeit ohne Vorab-Suche weiter
TIMEOUT_SECONDS = 10.0
# So viele frühere Nutzernachrichten bekommt das Intake als Kontext (z. B. für "Ab Garmisch" nach einer Rückfrage)
RECENT_USER_MESSAGES = 4

INSTRUCTIONS = """\
Du bereitest Anfragen an einen Outdoor-Tourenplaner vor. Bestimme die Orte, die der Planer für die NEUE Nachricht
des Nutzers per Geocoding (OpenStreetMap) nachschlagen muss, und nur diese:
- Tour: Start und Ziel (bei einer Rundtour nur der Start), ausdrücklich genannte Zwischenziele.
- Reine Bahnauskunft (Zug, Bahn, Bus, ÖPNV ohne Tour): Abfahrts- und Zielbahnhof. Nennt der Nutzer nur eine Stadt,
  hänge " Hbf" an (z. B. "Augsburg Hbf"); nennt er eine Adresse, übernimm sie.
- Mit Rad oder zu Fuß zum Bahnhof und dann mit dem Zug weiter: Start, Abfahrtsbahnhof UND Zielbahnhof. Der
  Abfahrtsbahnhof liegt im Startort, nicht am Ziel der Bahnfahrt. Beispiel: "Ich laufe von der Lindenstraße 5 in
  Münster zum Bahnhof und fahre dann mit dem Zug nach Köln" → "Lindenstraße 5, Münster", "Münster Hbf", "Köln Hbf".
- Start bzw. Ziel "zu Hause", "ab Haustür" o. Ä.: die Wohnadresse aus den Profil-Vorgaben, wörtlich.
Gib die Orte in der Reihenfolge der Reise aus (Start zuerst). Lass keinen Ort der Reise aus: Eine unvollständige
Liste führt den Planer in die Irre; lieber gar keine Orte als nur einen Teil.
Schreibe Ortsnamen so, wie man sie in einer Karte sucht ("Garmisch-Partenkirchen", "Zwischenahner Meer"), ohne
Zusätze wie "Bahnhof von". Setze `country_codes`, wenn das Land eindeutig ist (z. B. "de", "dk").

Gib eine leere Liste aus, wenn die Nachricht keinen Ort nennt oder offen lässt, welcher gemeint ist, und bei
Folgenachrichten, die nur einen bestehenden Plan ändern, ohne neue Orte zu nennen (z. B. "Tag 2 kürzer").
Orte, die laut Kontext schon nachgeschlagen wurden, nicht noch einmal ausgeben.
"""


class PlaceQuery(BaseModel):
    place_name: str = Field(description="Suchbegriff wie für geocode, z. B. \"Garmisch-Partenkirchen\"")
    country_codes: str | None = Field(default=None, description="Kommagetrennte ISO-Ländercodes, z. B. \"de\"")


class IntakeResult(BaseModel):
    places: list[PlaceQuery] = Field(description="Orte, die der Planer für diese Nachricht nachschlagen muss")


def build_intake_agent() -> Agent[None, IntakeResult]:
    settings = get_settings()
    return Agent(
        build_model(settings.intake_model, settings.llm_requests_per_minute) if settings.intake_model else None,
        name="intake",  # Name des Agent-Spans im Tracing
        output_type=ToolOutput(IntakeResult, name="orte", strict=True),
        instructions=INSTRUCTIONS,
        model_settings={"max_tokens": MAX_OUTPUT_TOKENS, "timeout": TIMEOUT_SECONDS, "max_retries": 0},
        defer_model_check=True,
    )


intake_agent = build_intake_agent()


def conversation_context(history_json: list[Any]) -> tuple[list[str], str | None, set[str]]:
    """Aus dem gespeicherten Verlauf (PydanticAI-JSON): letzte Nutzernachrichten, letzte Rückfrage des Planers und
    bereits nachgeschlagene Orte (klein geschrieben)."""
    user_messages: list[str] = []
    question: str | None = None
    geocoded: set[str] = set()
    for message in history_json:
        for part in message.get("parts", []):
            kind = part.get("part_kind")
            if kind == "user-prompt" and isinstance(part.get("content"), str):
                user_messages.append(part["content"])
            elif kind == "tool-call":
                args = part.get("args")
                if isinstance(args, str):
                    continue  # unvollständige Argumente; für den Kontext unwichtig
                if part.get("tool_name") == "ask_user":
                    question = (args or {}).get("question")
                elif part.get("tool_name") == "geocode" and (args or {}).get("place_name"):
                    geocoded.add(args["place_name"].strip().lower())
    return user_messages[-RECENT_USER_MESSAGES:], question, geocoded


def intake_prompt(message: str, history_json: list[Any], deps: AgentDeps) -> tuple[str, set[str]]:
    """Prompt für das Intake und die schon nachgeschlagenen Orte."""
    earlier, question, geocoded = conversation_context(history_json)
    lines = ["Profil-Vorgaben:"]
    lines.append(f"- Wohnadresse: {deps.home or 'keine'}")
    lines.append(f"- Standard-Fortbewegungsart: {deps.default_transport or 'keine'}")
    if earlier:
        lines.append("Frühere Nachrichten des Nutzers (älteste zuerst):")
        lines.extend(f"- {text}" for text in earlier)
    if question:
        lines.append(f"Letzte Rückfrage des Planers: {question}")
    if geocoded:
        lines.append("Schon nachgeschlagen: " + ", ".join(sorted(geocoded)))
    lines.append(f"NEUE Nachricht: {message}")
    return "\n".join(lines), geocoded


async def places_to_geocode(message: str, history_json: list[Any], deps: AgentDeps) -> list[PlaceQuery]:
    """Orte für die Vorab-Suche; ohne Intake-Modell leer. Fehler gibt der Aufrufer weiter (Graph fängt sie ab)."""
    if not get_settings().intake_model:
        return []
    prompt, geocoded = intake_prompt(message, history_json, deps)
    result = await intake_agent.run(prompt)
    unique: dict[tuple[str, str | None], PlaceQuery] = {}
    for place in result.output.places:
        name = place.place_name.strip()
        if name and name.lower() not in geocoded:
            unique.setdefault((name.lower(), place.country_codes), PlaceQuery(place_name=name, country_codes=place.country_codes))
    return list(unique.values())
