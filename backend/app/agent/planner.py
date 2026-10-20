from pydantic_ai import Agent, ModelRetry, ModelResponse, RunContext, Tool, ToolCallPart, ToolOutput

from app.agent.deps import AgentDeps
from app.agent.plan_schema import MAX_STAGE_KM, AdventurePlan, ClarificationQuestion
from app.agent.tools.accommodation import find_accommodation
from app.agent.tools.geocode import geocode
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.routing import estimate_hiking_time, plan_route
from app.agent.tools.sun import sun_times
from app.agent.tools.weather import get_weather
from app.config import get_settings

PlannerOutput = AdventurePlan | ClarificationQuestion

INSTRUCTIONS = """\
Du bist ein erfahrener Outdoor-Tourenplaner für Deutschland, Österreich, die Schweiz und das übrige Europa.
Du planst mit dem Nutzer ein "Abenteuer" und antwortest immer auf Deutsch.

## Pflichtangaben
Fortbewegungsart, Startdatum, Dauer in Tagen, Start, Ziel. Bei mehr als einem Tag zusätzlich die Übernachtungsart
(hut, camping, hotel, hostel, bivouac). Bei einer Rundtour ist das Ziel gleich dem Start.
Fehlt eine Pflichtangabe oder ist sie mehrdeutig, rufe `ask_user` auf: eine freundliche Rückfrage, die alle offenen
Punkte auf einmal klärt. Rate keine Pflichtangaben. Optionale Details (Fitness, Budget) darfst Du sinnvoll annehmen.
Relative Angaben wie "nächstes Wochenende" löst Du anhand des heutigen Datums auf.

## Vorgehen, sobald alles bekannt ist
Wähle die Tools passend zur Anfrage. Nicht jedes Tool ist immer nötig.
1. `geocode` für Start und Ziel (und ggf. Zwischenorte).
2. `plan_route` für die Gesamtstrecke. Bei mehrtägigen Touren die Strecke in sinnvolle Tagesetappen teilen.
   Richtwerte pro Tag: Wandern 10-25 km bzw. max. ~1200 Hm, Rad 40-100 km, MTB 25-60 km, Kanu 15-30 km,
   Auto max. ~500 km.
3. Mehrtägig: `find_accommodation` nahe den Etappenenden mit der gewünschten Übernachtungsart. Danach `plan_route`
   erneut mit den gewählten Unterkünften als Wegpunkte, damit die `segments` die echten Etappenwerte liefern.
4. `get_weather` für Ort und Zeitraum.
5. `search_knowledge` (Pflicht vor jedem Plan) für relevante Regeln und Tipps, z. B. Hüttenregeln, Camping und
   Wildcampen im jeweiligen Land, Ausrüstung für die Fortbewegungsart, Gewitter. Baue die Erkenntnisse in
   `packing_tips` und `safety_notes` ein.
6. Bei Wanderungen `estimate_hiking_time` je Etappe; bei langen Etappen `sun_times`.

## Regeln
- Verwende nur Fakten aus Tool-Ergebnissen. Unterkünfte ausschließlich aus `find_accommodation`; wurde keine
  gefunden, schreibe das in die Etappen-Notizen statt eine zu erfinden.
- Liefert ein Tool einen Fehler oder eine Schätzung, passe das Vorgehen an und weise im Plan darauf hin.
- Trage genutzte Wissensbasis-Dokumente (Titel) und Datendienste in `sources` ein.
- Etappenenden sind immer reale Orte (per `geocode` gefunden) oder Unterkünfte aus `find_accommodation`,
  niemals interpolierte Koordinaten. Wähle dafür passende Orte entlang der Strecke, z. B. Talorte oder Hüttenstandorte.
- Verwende im Plan kurze Ortsnamen (z. B. "Ulm" statt der vollständigen Geocoding-Adresse).
- Übernimm die `route_id` der finalen Route(n) in `route_ids`.
- Gibt der Nutzer später Änderungswünsche, passe den bestehenden Plan an und gib ihn vollständig neu aus.
- Gib das Ergebnis immer über `final_plan` oder `ask_user` aus.
"""


def used_tool(ctx: RunContext[AgentDeps], tool_name: str) -> bool:
    """Prüft den bisherigen Verlauf (inkl. früherer Nachrichten dieses Abenteuers) auf einen Tool-Aufruf."""
    return any(
        isinstance(part, ToolCallPart) and part.tool_name == tool_name
        for message in ctx.messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


def build_agent() -> Agent[AgentDeps, PlannerOutput]:
    agent = Agent(
        get_settings().llm_model,
        deps_type=AgentDeps,
        output_type=[
            ToolOutput(AdventurePlan, name="final_plan", description="Fertigen Abenteuerplan ausgeben"),
            ToolOutput(ClarificationQuestion, name="ask_user", description="Rückfrage an den Nutzer stellen"),
        ],
        instructions=INSTRUCTIONS,
        tools=[
            Tool(geocode),
            Tool(plan_route),
            Tool(estimate_hiking_time),
            Tool(get_weather),
            Tool(find_accommodation),
            Tool(search_knowledge),
            Tool(sun_times),
        ],
        retries=3,
        defer_model_check=True,
    )

    @agent.instructions
    def current_context(ctx: RunContext[AgentDeps]) -> str:
        return f"Heute ist {ctx.deps.today.isoformat()} ({ctx.deps.today:%A}). Der Nutzer heißt {ctx.deps.user_name}."

    @agent.output_validator
    def check_plausibility(ctx: RunContext[AgentDeps], output: PlannerOutput) -> PlannerOutput:
        """Selbstprüfung: Unplausible Pläne gehen mit konkretem Hinweis zurück an das Modell."""
        if not isinstance(output, AdventurePlan):
            return output
        if not used_tool(ctx, "search_knowledge"):
            raise ModelRetry(
                "Bevor Du den Plan ausgibst, durchsuche mit `search_knowledge` die Wissensbasis nach Regeln und Tipps "
                "zu Fortbewegungsart, Übernachtung und Region und berücksichtige sie im Plan."
            )
        if output.start_date < ctx.deps.today:
            raise ModelRetry(f"Das Startdatum {output.start_date} liegt in der Vergangenheit (heute: {ctx.deps.today}).")
        limit = MAX_STAGE_KM[output.transport]
        too_long = [s for s in output.stages if s.distance_km > limit]
        if too_long:
            days = ", ".join(f"Tag {s.day} ({s.distance_km:g} km)" for s in too_long)
            raise ModelRetry(
                f"Etappen zu lang für {output.transport.value} (max. {limit:g} km/Tag): {days}. "
                "Teile die Strecke anders auf oder frage den Nutzer nach mehr Tagen."
            )
        return output

    return agent


planner_agent = build_agent()
