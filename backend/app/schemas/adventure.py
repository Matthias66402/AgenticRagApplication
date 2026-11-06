from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import AdventureStatus


class AdventureCreate(BaseModel):
    title: str = Field(default="Neues Abenteuer", min_length=1, max_length=200)


class AdventureUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class AdventureSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    status: AdventureStatus
    created_at: datetime
    updated_at: datetime


class AdventureListItem(AdventureSummary):
    """Eintrag der Übersichtsliste; `places` für die Suche nach Start, Ziel, Etappenorten und Unterkünften."""

    places: list[str] = Field(default_factory=list)


def plan_places(plan: dict[str, Any] | None) -> list[str]:
    """Ortsnamen eines gespeicherten Plans, gekürzt auf den Teil vor dem ersten Komma, ohne Duplikate.

    Touren: Start, Ziel, Etappenorte und Unterkünfte. Reisekarten: Halte der Verbindungen und die Wege zum bzw. vom
    Bahnhof.
    """
    if not plan:
        return []
    places: list[dict[str, Any] | None] = [plan.get("start"), plan.get("destination")]
    for stage in plan.get("stages") or []:
        places += [stage.get("start"), stage.get("end"), stage.get("accommodation")]
    for leg in (plan.get("access"), plan.get("egress")):
        if leg:
            places += [leg.get("start"), leg.get("end")]
    names = [place.get("name") or "" for place in places if isinstance(place, dict)]
    for journey in (plan.get("outbound_journey"), plan.get("return_journey")):
        for leg in (journey or {}).get("legs", []):
            names += [leg.get("from_stop") or "", leg.get("to_stop") or ""]
    return list(dict.fromkeys(short for name in names if (short := name.split(",")[0].strip())))


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    tool_calls: list[Any]
    created_at: datetime


class AdventureRead(AdventureSummary):
    plan: dict[str, Any] | None
    messages: list[ChatMessageRead]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
