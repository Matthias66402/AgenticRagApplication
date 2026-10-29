import json
from typing import Any

from pydantic_ai import RunContext, ToolCallPart
from pydantic_ai.exceptions import UsageLimitExceeded

from app.agent.deps import AgentDeps
from app.agent.plan_schema import AccommodationType, TransportMode
from app.agent.research import research_agent
from app.agent.tools.common import tool_error
from app.rag.attribution import retrieved_documents


async def research_knowledge(
    ctx: RunContext[AgentDeps],
    region: str,
    transport: TransportMode,
    accommodation_type: AccommodationType | None = None,
    extra_topic: str | None = None,
) -> dict[str, Any]:
    """Lässt einen Recherche-Agenten die Wissensbasis (RAG) durchsuchen und die Treffer zusammenfassen.

    Liefert Hinweise zur Region, Regeln, Ausrüstung (`packing_tips`) und Sicherheit (`safety_notes`) sowie die
    Titel der gefundenen Dokumente (`documents`) für `sources`.

    Args:
        region: Region, Ort oder Route, z. B. "Oldenburg", "Harz" oder "Nordseeküsten-Radweg".
        transport: Fortbewegungsart der Tour.
        accommodation_type: Übernachtungsart bei mehrtägigen Touren.
        extra_topic: Höchstens ein Zusatzthema, z. B. "Fahrradmitnahme in der Bahn".
    """
    topics = [f"Region bzw. Route: {region}", f"Fortbewegungsart: {transport.value}"]
    if accommodation_type:
        topics.append(f"Übernachtungsart: {accommodation_type.value}")
    if extra_topic:
        topics.append(f"Zusatzthema: {extra_topic}")
    try:
        # Gemeinsame Nutzung: Die Anfragen des Recherche-Agenten zählen zum Limit des Planer-Laufs
        result = await research_agent.run("\n".join(topics), deps=ctx.deps, usage=ctx.usage)
    except UsageLimitExceeded:
        raise
    except Exception as exc:  # z. B. Modell nicht erreichbar oder keine gültige Zusammenfassung
        return tool_error(f"Recherche nicht möglich: {exc}. Ohne Wissensbasis planen und im Plan darauf hinweisen.")

    messages = result.all_messages()
    queries = [
        part.args_as_dict().get("query", "")
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "search_knowledge"
    ]
    documents = retrieved_documents(json.loads(result.all_messages_json()))
    if not documents:
        return tool_error(
            f"Die Wissensbasis enthält dazu nichts Passendes (gesucht: {', '.join(queries) or '–'}). "
            "Ohne diese Informationen planen und im Plan darauf hinweisen."
        )
    return result.output.model_dump() | {"queries": queries, "documents": documents}
