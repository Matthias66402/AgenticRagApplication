"""Intake: Ein kleines LLM bestimmt vorab, was der Planer für die aktuelle Nachricht sicher braucht: die Orte, die er
nachschlagen muss, und bei einer neuen Tour den Auftrag für die Recherche in der Wissensbasis.

Der Graph (app.agent.graph) führt `geocode` und `research_knowledge` damit parallel aus, bevor der Planer startet,
und fügt Aufrufe und Ergebnisse in dessen Verlauf ein. Das spart dem Planer Runden, und die vor jedem Plan
verlangte Recherche steht sicher im Verlauf. Ein Fehler hier ist unkritisch: Der Planer kann jedes Tool selbst
(erneut) aufrufen.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.models import build_model
from app.agent.plan_schema import AccommodationType, TransportMode
from app.config import get_settings

# Recherche-Tools im Verlauf; search_knowledge: frühere Abenteuer, in denen der Planer noch selbst gesucht hat
RESEARCH_TOOLS = {"research_knowledge", "search_knowledge"}

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

Recherche (`research`): Vor jedem neuen Tourenplan durchsucht ein Recherche-Agent die Wissensbasis. Fülle
`research` nur, wenn der Nutzer eine Tour (Wandern, Rad, MTB, Kanu, Auto) plant und sowohl Region bzw. Route als
auch Fortbewegungsart feststehen (auch aus dem Kontext oder der Standard-Fortbewegungsart des Profils); sonst null,
insbesondere bei reiner Bahnauskunft, bei Rad bzw. Fußweg nur zum Bahnhof und wenn die Region noch offen ist.
- `region`: Ortsname bzw. Name des Fernwegs, z. B. "Oldenburg", "Harz" oder "Nordseeküsten-Radweg" (bei einer Tour
  entlang einer Küste oder eines Flusses der passende Fernweg, z. B. "entlang der Elbe" → "Elberadweg").
- `accommodation_type`: bei mehrtägigen Touren die gewünschte Übernachtungsart (bei mehreren die zuerst genannte),
  sonst null. Zuordnung: Shelter, Biwak, Zelt in der Natur, Wildcampen → "bivouac"; Campingplatz → "camping";
  "hut" nur, wenn der Nutzer ausdrücklich Hütten nennt (Berghütte, Alpenvereinshütte, Hüttenwanderung).
- `extra_topic`: höchstens ein Zusatzthema, sonst null. Bei Rad- oder MTB-Touren mit An- oder Abreise per Bahn
  "Fahrradmitnahme in der Bahn". Bei Shelter- bzw. Wildcamping-Übernachtung "Wildcampen in <Land>" mit dem
  ausgeschriebenen Namen des Landes, in dem die Tour liegt, abgeleitet aus ihren Orten (z. B. eine Tour von Bergen
  nach Ålesund → "Wildcampen in Norwegen"), nicht aus der Sprache der Nachricht.
"""


class PlaceQuery(BaseModel):
    place_name: str = Field(description="Suchbegriff wie für geocode, z. B. \"Garmisch-Partenkirchen\"")
    country_codes: str | None = Field(default=None, description="Kommagetrennte ISO-Ländercodes, z. B. \"de\"")


class ResearchRequest(BaseModel):
    """Argumente für research_knowledge (app.agent.tools.research)."""

    region: str = Field(description="Region, Ort oder Route, z. B. \"Harz\" oder \"Nordseeküsten-Radweg\"")
    transport: TransportMode
    accommodation_type: AccommodationType | None = None
    extra_topic: str | None = Field(default=None, description="Höchstens ein Zusatzthema")


class IntakeResult(BaseModel):
    places: list[PlaceQuery] = Field(description="Orte, die der Planer für diese Nachricht nachschlagen muss")
    research: ResearchRequest | None = Field(description="Recherche für eine neue Tour, sonst null")


class Prefetch(BaseModel):
    """Was der Graph vor dem Planer ausführt."""

    places: list[PlaceQuery] = Field(default_factory=list)
    research: ResearchRequest | None = None


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


@dataclass
class ConversationContext:
    """Aus dem gespeicherten Verlauf (PydanticAI-JSON) für das Intake."""

    user_messages: list[str] = field(default_factory=list)  # letzte Nutzernachrichten, älteste zuerst
    question: str | None = None  # letzte Rückfrage des Planers
    geocoded: set[str] = field(default_factory=set)  # bereits nachgeschlagene Orte, klein geschrieben
    researched: bool = False  # Wissensbasis schon durchsucht


def conversation_context(history_json: list[Any]) -> ConversationContext:
    context = ConversationContext()
    for message in history_json:
        for part in message.get("parts", []):
            kind = part.get("part_kind")
            if kind == "user-prompt" and isinstance(part.get("content"), str):
                context.user_messages.append(part["content"])
            elif kind == "tool-call":
                tool_name = part.get("tool_name")
                context.researched |= tool_name in RESEARCH_TOOLS
                args = part.get("args")
                if not isinstance(args, dict):
                    continue  # unvollständige Argumente; für den Kontext unwichtig
                if tool_name == "ask_user":
                    context.question = args.get("question")
                elif tool_name == "geocode" and args.get("place_name"):
                    context.geocoded.add(args["place_name"].strip().lower())
    context.user_messages = context.user_messages[-RECENT_USER_MESSAGES:]
    return context


def intake_prompt(message: str, context: ConversationContext, deps: AgentDeps) -> str:
    earlier, question, geocoded = context.user_messages, context.question, context.geocoded
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
    return "\n".join(lines)


async def plan_prefetch(message: str, history_json: list[Any], deps: AgentDeps) -> Prefetch:
    """Was vor dem Planer laufen soll; ohne Intake-Modell nichts. Fehler gibt der Aufrufer weiter (Graph fängt sie ab).

    Bereits nachgeschlagene Orte und eine Recherche, die schon im Verlauf steht, filtert der Code heraus, nicht das LLM.
    """
    if not get_settings().intake_model:
        return Prefetch()
    context = conversation_context(history_json)
    result = await intake_agent.run(intake_prompt(message, context, deps))
    unique: dict[tuple[str, str | None], PlaceQuery] = {}
    for place in result.output.places:
        name = place.place_name.strip()
        if name and name.lower() not in context.geocoded:
            key = (name.lower(), place.country_codes)
            unique.setdefault(key, PlaceQuery(place_name=name, country_codes=place.country_codes))
    research = result.output.research
    if research is not None and (context.researched or not research.region.strip()):
        research = None
    return Prefetch(places=list(unique.values()), research=research)
