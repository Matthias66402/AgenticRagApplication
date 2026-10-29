"""Recherche-Agent: durchsucht die Wissensbasis im Auftrag des Planers und fasst die Treffer zusammen.

Der Planer ruft ihn über das Tool `research_knowledge` auf ("Agent als Tool", app.agent.tools.research). Die
Suchen, Fundstellen und Umformulierungen bleiben im Verlauf dieses Agenten; der Planer bekommt nur die kurze
Zusammenfassung und hält so seinen Kontext klein.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.models import build_model, model_settings
from app.agent.tools.knowledge import search_knowledge
from app.config import get_settings

INSTRUCTIONS = """\
Du recherchierst für einen Outdoor-Tourenplaner in seiner Wissensbasis und antwortest auf Deutsch.

Stelle mit `search_knowledge` kurze Suchen zu je einem Thema, alle in derselben Runde:
1. Region bzw. Route mit Ortsnamen, z. B. "Radrouten rund um Oldenburg" oder "Wandern im Harz".
2. Ausrüstung und Sicherheit für die Fortbewegungsart, z. B. "Ausrüstung Radtour" oder "Ausrüstung Tageswanderung".
3. Falls angegeben: Übernachtungsart (z. B. "Hüttenregeln und Reservierung", "Wildcampen in Norwegen") und das
   Zusatzthema (z. B. "Fahrradmitnahme in der Bahn").
Findet eine Suche nichts Passendes, darfst Du sie einmal anders formulieren. Mehr als 5 Suchen sind nicht nötig.

Fasse danach in kurzen Stichpunkten zusammen, was die Fundstellen für die Tour hergeben. Übernimm nur, was in den
Fundstellen steht, und erfinde nichts. Gibt die Wissensbasis zu einem Punkt nichts her, bleibt die Liste leer.
"""


class ResearchBrief(BaseModel):
    region_notes: list[str] = Field(description="Hinweise zu Region bzw. Route: Wegverlauf, Highlights, Besonderheiten")
    rules: list[str] = Field(description="Regeln, z. B. zu Übernachtung, Wildcampen, Fahrradmitnahme, Reservierung")
    packing_tips: list[str] = Field(description="Ausrüstung für Fortbewegungsart und Übernachtung")
    safety_notes: list[str] = Field(description="Sicherheitshinweise, z. B. Wetter, Gelände, Verkehr")


def build_research_agent() -> Agent[AgentDeps, ResearchBrief]:
    settings = get_settings()
    return Agent(
        build_model(settings.research_model, settings.llm_requests_per_minute),
        name="knowledge-researcher",  # Name des Agent-Spans im Tracing, unterhalb des Planers
        deps_type=AgentDeps,
        output_type=ToolOutput(
            ResearchBrief, name="research_brief", description="Zusammenfassung der Recherche ausgeben", strict=True
        ),
        model_settings=model_settings(settings.research_model),
        instructions=INSTRUCTIONS,
        tools=[Tool(search_knowledge)],
        retries=2,
        defer_model_check=True,
    )


research_agent = build_research_agent()
