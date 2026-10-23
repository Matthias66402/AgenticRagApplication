import datetime as dt
import re
from typing import Any

from pydantic_ai import (
    Agent, ModelRequest, ModelResponse, ModelRetry, RunContext, Tool, ToolCallPart, ToolOutput, ToolReturnPart,
    UserPromptPart,
)  # fmt: skip
from pydantic_ai.models import Model
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.settings import ModelSettings

from app.agent.deps import AgentDeps
from app.agent.plan_schema import (
    MAX_STAGE_KM, AccessRoute, AdventurePlan, ClarificationQuestion, Journey, TextAnswer, TrainTripPlan,
)  # fmt: skip
from app.agent.rate_limit import RateLimitedModel
from app.agent.tools.accommodation import find_accommodation
from app.agent.tools.geocode import geocode
from app.agent.tools.knowledge import search_knowledge
from app.agent.tools.routing import estimate_hiking_time, plan_route
from app.agent.tools.sun import sun_times
from app.agent.tools.transit import find_train_connections
from app.agent.tools.weather import get_weather
from app.config import get_settings

PlannerOutput = AdventurePlan | TrainTripPlan | ClarificationQuestion | TextAnswer

# Bezeichnungen der Profil-Vorgabe für die Anweisungen (app.schemas.user.DefaultTransport)
TRANSPORT_LABELS = {"hiking": "Wandern", "cycling": "Radtour"}
# Hinweis auf eine gewünschte Fahrt mit öffentlichen Verkehrsmitteln (nicht in "Fahrbahn", "Fährverbindung" o. Ä.)
PUBLIC_TRANSPORT_WORDS = re.compile(
    r"\b(zug|züge[n]?|bahn|bahnhof|s-bahn|bus|busse[n]?|öpnv|nahverkehr|ice|ic|rb|re|regionalzug|regionalbahn)\b",
    re.IGNORECASE,
)

# Verweis auf angehängte Verbindungen in einer Antwort ("siehst du unten", "darunter angezeigt")
SHOWN_BELOW = re.compile(r"\b(unten|darunter)\b", re.IGNORECASE)

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
    auflöst (z. B. in Tests ohne API-Key)."""
    return RateLimitedModel(model, requests_per_minute) if requests_per_minute > 0 else model


INSTRUCTIONS = """\
Du bist ein erfahrener Outdoor-Tourenplaner für Deutschland, Österreich, die Schweiz und das übrige Europa.
Du planst mit dem Nutzer ein "Abenteuer" und antwortest immer auf Deutsch.

## Pflichtangaben
Fortbewegungsart, Startdatum, Dauer in Tagen, Start, Ziel. Bei mehr als einem Tag zusätzlich die Übernachtungsart
(hut, camping, hotel, hostel, bivouac). Bei einer Rundtour ist das Ziel gleich dem Start.
Nennt der Nutzer keine Fortbewegungsart bzw. keinen Start, gelten seine Profil-Vorgaben (siehe unten, falls
vorhanden). Ein allgemeines "fahren" ohne Verkehrsmittel ist dann eine Tour mit der Standard-Fortbewegungsart.
Fehlt eine Pflichtangabe oder ist sie mehrdeutig, rufe `ask_user` auf: eine freundliche Rückfrage, die alle offenen
Punkte auf einmal klärt. Rate keine Pflichtangaben. Optionale Details (Fitness, Budget) darfst Du sinnvoll annehmen.
Relative Angaben wie "nächstes Wochenende" löst Du anhand des heutigen Datums auf.

## Reine Bahnauskunft
Nur wenn der Nutzer ausdrücklich Zug, Bahn, Bus bzw. ÖPNV nennt und keine Tour möchte. Sätze wie "nach Hause
fahren" oder "von A nach B fahren" ohne Verkehrsmittel sind keine Bahnauskunft, wenn er eine
Standard-Fortbewegungsart hat: dann eine Tour damit planen. Bei einer Bahnauskunft erstelle keinen Abenteuerplan:
`geocode` für beide Bahnhöfe (z. B. "Augsburg Hbf", sofern der Nutzer keine Adresse nennt; fehlen Datum oder
Uhrzeit, per `ask_user` nachfragen), einmal `find_train_connections`, dann `answer_user` mit einer kurzen Einordnung
(z. B. schnellste Verbindung, wenige Umstiege). Die Verbindungen werden automatisch unter Deiner Antwort angezeigt;
liste sie nicht erneut auf. Das gilt nur für Abfragen in derselben Nachricht: Ohne neuen Aufruf von
`find_train_connections` erscheint nichts unter der Antwort. Biete höchstens an, was Deine Tools können (andere
Uhrzeit, Rückfahrt, Reisekarte zu einer Verbindung, eine Tour am Zielort), keine Buchung, Favoriten oder Suche nach
Fahrradmitnahme.
Wählt der Nutzer eine Verbindung aus oder möchte er einen Plan bzw. eine Übersicht seiner Bahnfahrt, gib eine
Reisekarte über `final_train_trip` aus: die gewählte Verbindung unverändert aus dem `find_train_connections`-Ergebnis
(auch aus früheren Nachrichten) übernehmen, eine Rückfahrt nur, wenn sie gesucht wurde und gewünscht ist. Bittet er
schon in der ersten Nachricht um einen Plan, wähle selbst die passendste Verbindung (bei einer gewünschten
Ankunftszeit die späteste, die rechtzeitig ankommt) und frage nicht nach.
Möchte der Nutzer mit dem Rad oder zu Fuß zum Bahnhof und dann mit dem Zug weiter (ohne Tour am Ziel), gehört dieser
Weg in die Reisekarte (`access`): `geocode` für Wohnadresse bzw. Start und Bahnhof, `plan_route` dorthin
(transport cycling bzw. hiking), `find_train_connections` ab dem Bahnhof, dann `final_train_trip`. Losfahren
(`access.departure`) = Abfahrt der Verbindung minus Fahr- bzw. Gehzeit minus Puffer (Rad abstellen: 10 Minuten,
zu Fuß: 5 Minuten).

## Vorgehen, sobald alles bekannt ist
Wähle die Tools passend zur Anfrage. Nicht jedes Tool ist immer nötig. Rufe voneinander unabhängige Tools im
selben Schritt gemeinsam auf statt nacheinander, z. B. `geocode` für alle Orte, die beiden Pflichtsuchen,
`get_weather` zusammen mit `find_accommodation` und `estimate_hiking_time` für alle Etappen.
1. `geocode` für Start und Ziel (und ggf. Zwischenorte).
2. `plan_route` für die Gesamtstrecke. Bei mehrtägigen Touren die Strecke in sinnvolle Tagesetappen teilen.
   Richtwerte pro Tag: Wandern 10-25 km bzw. max. ~1200 Hm, Rad 40-100 km, MTB 25-60 km, Kanu 15-30 km,
   Auto max. ~500 km.
   Soll die Tour einem ausgeschilderten Fernweg folgen (z. B. "auf dem Ostseeküsten-Radweg"), übergib dessen Namen
   bei jedem `plan_route` dieser Tour als `route_name` und wähle Etappenorte und Unterkünfte nahe am Fernweg
   (max. 5 km). Findet das Tool den Namen nicht, wiederhole mit dem passenden Namen aus der genannten Liste. Der
   Plan nutzt dann diese Route; passt sie nicht in die gewünschten Tage, frage nach mehr Tagen statt abzukürzen.
3. Mehrtägig: `find_accommodation` nahe den Etappenenden mit der gewünschten Übernachtungsart. Danach `plan_route`
   erneut mit den gewählten Unterkünften als Wegpunkte, damit die `segments` die echten Etappenwerte liefern.
4. `get_weather` für Ort und Zeitraum.
5. `search_knowledge` mit kurzen Suchen zu je einem Thema statt einer Sammelanfrage. Vor jedem neuen Plan
   sind diese beiden Suchen Pflicht:
   a) Region bzw. Route mit Ortsnamen, z. B. "Radrouten rund um Oldenburg" oder "Wandern im Harz",
   b) Ausrüstung und Sicherheit für die Fortbewegungsart, z. B. "Ausrüstung Radtour" oder "Ausrüstung
      Tageswanderung". Daraus stammen `packing_tips` und `safety_notes`.
   Höchstens eine weitere Suche, z. B. Fahrradmitnahme (Schritt 7), Hüttenregeln oder Wildcampen im jeweiligen
   Land; sie ersetzt keine der Pflichtsuchen. Passt der Nutzer nur einen bestehenden Plan an, entfallen Suchen,
   deren Ergebnisse schon im Verlauf stehen. Hinweise zur Region gehören auch in die Etappen-Notizen.
6. Bei Wanderungen `estimate_hiking_time` je Etappe; bei langen Etappen `sun_times`.
7. Möchte der Nutzer mit Bahn bzw. öffentlichen Verkehrsmitteln an- oder abreisen, `find_train_connections`:
   Anreise vom Wohn- bzw. Abfahrtsort zum Tourstart am ersten Tag (`arrive_by`, Ankunft mit genug Zeit für die
   erste Etappe), Rückreise vom Tourziel am letzten Tag (Abfahrt nach dem geschätzten Etappenende). Übernimm je eine
   passende Verbindung in `outbound_journey` bzw. `return_journey` und berücksichtige die Ankunftszeit bei der
   ersten Etappe. Fehlt der Abfahrtsort, frage danach. Bei Rad- und MTB-Touren mit `search_knowledge` nach Regeln
   zur Fahrradmitnahme suchen (die weitere Suche aus Schritt 5) und darauf hinweisen.

## Regeln
- Verwende nur Fakten aus Tool-Ergebnissen. Unterkünfte ausschließlich aus `find_accommodation`; wurde keine
  gefunden, schreibe das in die Etappen-Notizen statt eine zu erfinden.
- Liefert ein Tool einen Fehler oder eine Schätzung, passe das Vorgehen an und weise im Plan darauf hin.
- Bahnverbindungen ausschließlich aus `find_train_connections` übernehmen (Zeiten, Linien, Gleise unverändert).
  Ändern sich Abfahrts- oder Zielort einer Bahnfahrt (z. B. andere Unterkunft), die Verbindung neu abfragen.
- Trage genutzte Wissensbasis-Dokumente (Titel) und Datendienste in `sources` ein.
- Etappenenden sind immer reale Orte (per `geocode` gefunden) oder Unterkünfte aus `find_accommodation`,
  niemals interpolierte Koordinaten. Wähle dafür passende Orte entlang der Strecke, z. B. Talorte oder Hüttenstandorte.
- Verwende im Plan kurze Ortsnamen (z. B. "Ulm" statt der vollständigen Geocoding-Adresse).
- Übernimm die `route_id` der finalen Route(n) in `route_ids`.
- Gibt der Nutzer später Änderungswünsche, passe den bestehenden Plan an und gib ihn vollständig neu aus.
- Rufe ein Tool nicht mehrfach mit denselben Argumenten auf; das Ergebnis ändert sich nicht.
- Gib das Ergebnis immer über `final_plan`, `ask_user` oder (nur ohne Tour) `final_train_trip` bzw. `answer_user`
  aus. Sobald alle Pflichtangaben einer Tour bekannt sind, gib den Plan aus, statt nur Eckdaten zu nennen oder
  nachzufragen, ob ein Plan gewünscht ist.
"""


def used_tool(ctx: RunContext[AgentDeps], tool_name: str) -> bool:
    """Prüft den bisherigen Verlauf (inkl. früherer Nachrichten dieses Abenteuers) auf einen Tool-Aufruf."""
    return any(
        isinstance(part, ToolCallPart) and part.tool_name == tool_name
        for message in ctx.messages
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


def mentions_public_transport(ctx: RunContext[AgentDeps]) -> bool:
    """Hat der Nutzer in einer seiner Nachrichten Zug, Bahn, Bus oder ÖPNV erwähnt?"""
    return any(
        PUBLIC_TRANSPORT_WORDS.search(part.content)
        for message in ctx.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def check_default_transport(ctx: RunContext[AgentDeps], output: PlannerOutput) -> None:
    """Mit Standard-Fortbewegungsart im Profil wird ohne genanntes Verkehrsmittel weder danach gefragt noch nur eine
    Bahnauskunft gegeben (das Modell deutet z. B. "nach Hause fahren" sonst gern als Zugfahrt)."""
    transport = ctx.deps.default_transport
    if not transport or mentions_public_transport(ctx):
        return
    label = TRANSPORT_LABELS.get(transport, transport)
    if isinstance(output, ClarificationQuestion) and "transport" in output.missing_fields:
        raise ModelRetry(
            f"Der Nutzer hat kein Verkehrsmittel genannt, seine Standard-Fortbewegungsart ist {label} "
            f"(transport={transport}). Frage nicht danach, sondern plane damit; frage höchstens nach anderen "
            "fehlenden Pflichtangaben."
        )
    if isinstance(output, (TextAnswer, TrainTripPlan)) and used_tool(ctx, "find_train_connections"):
        raise ModelRetry(
            f"Der Nutzer hat weder Zug noch Bahn genannt, also ist das keine Bahnauskunft. Plane eine Tour mit seiner "
            f"Standard-Fortbewegungsart {label} (transport={transport})."
        )


def found_connections(ctx: RunContext[AgentDeps]) -> dict[tuple[str, str], dict[str, Any]]:
    """Verbindungen aus `find_train_connections` im Verlauf (inkl. früherer Nachrichten) nach Abfahrt und Ankunft."""
    return {
        (connection["departure"], connection["arrival"]): connection
        for message in ctx.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "find_train_connections"
        and isinstance(part.content, dict)
        for connection in part.content.get("connections", [])
    }  # fmt: skip


def as_found(journey: Journey | None, known: dict[tuple[str, str], dict[str, Any]]) -> Journey | None:
    """Teilstrecken unverändert aus dem Fahrplanergebnis, damit keine Linie (z. B. "Bus 24") verloren geht."""
    if journey is None or (connection := known.get((journey.departure, journey.arrival))) is None:
        return journey
    return Journey.model_validate(connection | {"notes": journey.notes})


def check_journeys_found(
    ctx: RunContext[AgentDeps], known: dict[tuple[str, str], dict[str, Any]], journeys: dict[str, Journey | None]
) -> None:
    """Plan und Reisekarte zeigen nur Verbindungen, die die Fahrplanauskunft tatsächlich geliefert hat."""
    for label, journey in journeys.items():
        if journey is None:
            continue
        if (journey.departure, journey.arrival) not in known:
            raise ModelRetry(
                f"Die {label} ({journey.departure} → {journey.arrival}) stammt aus keinem `find_train_connections`-"
                "Ergebnis. Übernimm eine gefundene Verbindung unverändert, frage die Verbindung neu ab oder lass sie leer, "
                "wenn keine gefunden wurde."
            )
        if journey.departure[:10] < ctx.deps.today.isoformat():
            raise ModelRetry(f"Die {label} liegt in der Vergangenheit (heute: {ctx.deps.today}).")


def check_train_trip(ctx: RunContext[AgentDeps], output: TrainTripPlan) -> TrainTripPlan:
    known = found_connections(ctx)
    check_journeys_found(ctx, known, {"Hinfahrt": output.outbound_journey, "Rückfahrt": output.return_journey})
    if output.access is not None:
        check_access(ctx, output.access, output.outbound_journey)
    return output.model_copy(
        update={
            "outbound_journey": as_found(output.outbound_journey, known),
            "return_journey": as_found(output.return_journey, known),
        }
    )


def check_access(ctx: RunContext[AgentDeps], access: AccessRoute, journey: Journey) -> None:
    """Der Weg zum Bahnhof braucht eine berechnete Route und muss vor Abfahrt der Verbindung enden."""
    if access.route_id not in ctx.deps.routes:
        raise ModelRetry(
            f"Die route_id {access.route_id!r} stammt aus keinem `plan_route`-Ergebnis. Berechne den Weg zum Bahnhof "
            "mit `plan_route` und übernimm dessen route_id, Distanz und Dauer."
        )
    arrival = dt.datetime.fromisoformat(access.departure) + dt.timedelta(hours=access.duration_hours)
    if arrival > dt.datetime.fromisoformat(journey.departure):
        raise ModelRetry(
            f"Mit Losfahren um {access.departure[11:]} und {access.duration_hours:g} h Fahrzeit ist der Nutzer erst "
            f"{arrival:%H:%M} am Bahnhof, die Verbindung fährt aber {journey.departure[11:]} ab. Rechne "
            "`access.departure` neu (Abfahrt minus Fahrzeit minus Puffer)."
        )


def check_named_route_used(ctx: RunContext[AgentDeps], output: AdventurePlan) -> None:
    """Wurde eine Route entlang eines Fernwegs berechnet, darf der Plan keine ältere Route nutzen.

    Sonst weicht das Modell gern auf eine kürzere Route ohne Fernweg aus, wenn die Strecke nicht in die Tage passt.
    """
    results = [
        part.content
        for message in ctx.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "plan_route"
        and isinstance(part.content, dict) and "route_id" in part.content
    ]  # fmt: skip
    order = {result["route_id"]: index for index, result in enumerate(results)}
    named = [index for index, result in enumerate(results) if "named_route" in result]
    used = [order[route_id] for route_id in output.route_ids if route_id in order]
    if not named or not used or max(used) >= named[-1]:
        return
    latest = results[named[-1]]
    raise ModelRetry(
        f"Der Plan nutzt eine ältere Route statt der Route entlang {', '.join(latest['named_route']['routes'])} "
        f"(route_id {latest['route_id']}, {latest['distance_km']:g} km). Übernimm diese route_id und die Etappenwerte "
        "aus ihren segments. Passt die Strecke nicht in die gewünschten Tage, kürze nicht über eine andere Route ab, "
        "sondern frage mit `ask_user` nach mehr Tagen oder einem näheren Ziel."
    )


def check_connections_shown(ctx: RunContext[AgentDeps], output: TextAnswer) -> None:
    """Verbindungen hängt das Backend nur an, wenn sie in diesem Lauf abgefragt wurden (siehe runner)."""
    if not ctx.deps.train_queries and SHOWN_BELOW.search(output.message):
        raise ModelRetry(
            "Unter Deiner Antwort wird nichts angezeigt, weil Du in dieser Nachricht keine Verbindung abgefragt hast. "
            "Verweise nicht auf \"unten\". Möchte der Nutzer eine Verbindung sehen, gib eine Reisekarte über "
            "`final_train_trip` aus oder frage die Verbindungen mit `find_train_connections` erneut ab."
        )


def build_agent() -> Agent[AgentDeps, PlannerOutput]:
    agent = Agent(
        build_model(get_settings().llm_model, get_settings().llm_requests_per_minute),
        name="adventure-planner",  # Name des Agent-Spans im Tracing
        deps_type=AgentDeps,
        output_type=[
            ToolOutput(AdventurePlan, name="final_plan", description="Fertigen Abenteuerplan ausgeben", strict=True),
            ToolOutput(
                ClarificationQuestion, name="ask_user", description="Rückfrage an den Nutzer stellen", strict=True
            ),
            ToolOutput(
                TrainTripPlan,
                name="final_train_trip",
                description="Reisekarte für eine reine Bahnfahrt ohne Tour ausgeben, z. B. zur gewählten Verbindung",
                strict=True,
            ),
            ToolOutput(
                TextAnswer,
                name="answer_user",
                description="Antwort ohne Abenteuerplan, z. B. reine Bahnauskunft",
                strict=True,
            ),
        ],
        model_settings=model_settings(get_settings().llm_model, get_settings().llm_reasoning_effort),
        instructions=INSTRUCTIONS,
        tools=[
            Tool(geocode),
            Tool(plan_route),
            Tool(estimate_hiking_time),
            Tool(get_weather),
            Tool(find_accommodation),
            Tool(search_knowledge),
            Tool(sun_times),
            Tool(find_train_connections),
        ],
        retries=3,
        defer_model_check=True,
    )

    @agent.instructions
    def current_context(ctx: RunContext[AgentDeps]) -> str:
        deps = ctx.deps
        lines = [f"Heute ist {deps.today.isoformat()} ({deps.today:%A}). Der Nutzer heißt {deps.user_name}."]
        if deps.home or deps.default_transport:
            lines.append("Profil-Vorgaben des Nutzers (gelten, wenn er nichts anderes nennt; frage dann nicht danach):")
        if deps.home:
            lines.append(
                f"- Wohnadresse: {deps.home}. Start bzw. Abfahrtsort und Ziel bei \"nach Hause\" (per `geocode` suchen)."
            )
        if deps.default_transport:
            label = TRANSPORT_LABELS.get(deps.default_transport, deps.default_transport)
            lines.append(f"- Standard-Fortbewegungsart: {label} (transport={deps.default_transport}).")
        return "\n".join(lines)

    @agent.output_validator
    def check_plausibility(ctx: RunContext[AgentDeps], output: PlannerOutput) -> PlannerOutput:
        """Selbstprüfung: Unplausible Pläne gehen mit konkretem Hinweis zurück an das Modell."""
        check_default_transport(ctx, output)
        if isinstance(output, TextAnswer):
            check_connections_shown(ctx, output)
        if isinstance(output, TrainTripPlan):
            return check_train_trip(ctx, output)
        if not isinstance(output, AdventurePlan):
            return output
        if not used_tool(ctx, "search_knowledge"):
            raise ModelRetry(
                "Bevor Du den Plan ausgibst, durchsuche mit `search_knowledge` die Wissensbasis nach Regeln und Tipps "
                "zu Fortbewegungsart, Übernachtung und Region und berücksichtige sie im Plan."
            )
        if output.start_date < ctx.deps.today:
            raise ModelRetry(f"Das Startdatum {output.start_date} liegt in der Vergangenheit (heute: {ctx.deps.today}).")
        check_named_route_used(ctx, output)
        if (output.outbound_journey or output.return_journey) and not used_tool(ctx, "find_train_connections"):
            raise ModelRetry(
                "An- und Rückreise nur mit Verbindungen aus `find_train_connections` angeben. "
                "Rufe das Tool auf oder lass outbound_journey/return_journey leer."
            )
        known = found_connections(ctx)
        check_journeys_found(ctx, known, {"Anreise": output.outbound_journey, "Rückreise": output.return_journey})
        output = output.model_copy(
            update={
                "outbound_journey": as_found(output.outbound_journey, known),
                "return_journey": as_found(output.return_journey, known),
            }
        )
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
