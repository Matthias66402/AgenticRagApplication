from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class AdventureStatus(StrEnum):
    draft = "draft"
    planned = "planned"


class Adventure(Base):
    __tablename__ = "adventures"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="Neues Abenteuer")
    status: Mapped[str] = mapped_column(String(20), default=AdventureStatus.draft)
    # Validierter AdventurePlan (siehe app.agent.plan_schema)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Vollständiger PydanticAI-Nachrichtenverlauf, damit der Agent Folgeeingaben im Kontext versteht
    agent_history: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    # Routengeometrien aus Läufen ohne Plan (z. B. Route berechnet, dann Rückfrage): route_id -> [[lat, lon, ele], ...].
    # Der nächste Plan kann sich auf sie beziehen; mit dem Plan werden sie übernommen und hier geleert.
    pending_routes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="adventure", cascade="all, delete-orphan", order_by="ChatMessage.id"
    )


class ChatMessage(Base):
    """Für die Anzeige aufbereiteter Chatverlauf inkl. der Tool-Aufrufe des Agenten."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    adventure_id: Mapped[int] = mapped_column(ForeignKey("adventures.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    adventure: Mapped[Adventure] = relationship(back_populates="messages")
