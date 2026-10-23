from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, deferred, mapped_column

from app.db import Base


class LongDistanceRoute(Base):
    """Vorab importierter Verlauf eines Fernwegs aus OpenStreetMap (siehe app.seed.long_distance_routes).

    Live-Abfragen bei Overpass dauern oft ein bis zwei Minuten oder scheitern; importierte Fernwege stehen beim
    Planen sofort bereit.
    """

    __tablename__ = "long_distance_routes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    # Weitere Namen, unter denen Nutzer den Fernweg kennen, z. B. "Ostseeradweg", "D2", "EuroVelo 10"
    aliases: Mapped[list[str]] = mapped_column(JSONB, default=list)
    # OSM-Routentyp: bicycle oder hiking
    route_type: Mapped[str] = mapped_column(String(20))
    relation_ids: Mapped[list[int]] = mapped_column(JSONB, default=list)
    # Wege der Relationen: [[Knoten-IDs], [[lat, lon], ...]]; groß, daher nur bei Bedarf geladen
    ways: Mapped[list[Any]] = deferred(mapped_column(JSONB, default=list, nullable=False))
    length_km: Mapped[float] = mapped_column(default=0.0)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
