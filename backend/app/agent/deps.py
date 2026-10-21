from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings


@dataclass
class AgentDeps:
    """Abhängigkeiten, die jedes Tool über ``ctx.deps`` erhält."""

    settings: Settings
    http: httpx.AsyncClient
    # Factory statt Session: Tools können parallel laufen, eine AsyncSession ist nicht nebenläufig nutzbar.
    session_factory: async_sessionmaker[AsyncSession]
    user_name: str
    today: date = field(default_factory=date.today)
    # route_id -> Geometrie [[lat, lon, ele], ...]; für Karte und GPX, wird nicht an das LLM zurückgegeben
    routes: dict[str, list[list[float]]] = field(default_factory=dict)
    # Fahrplanabfragen dieses Laufs: Schlüssel aus den Suchparametern -> Abfrage samt Verbindungen
    train_queries: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
