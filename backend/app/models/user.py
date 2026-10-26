from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class UserRole(StrEnum):
    user = "user"
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(100))
    # Optionale Vorgaben für die Planung: Wohnadresse als Start bzw. Abfahrtsort, Fortbewegungsart ohne Angabe
    home_address: Mapped[str | None] = mapped_column(String(200))  # Straße und Hausnummer
    home_city: Mapped[str | None] = mapped_column(String(100))  # PLZ und Ort
    default_transport: Mapped[str | None] = mapped_column(String(20))
    # Mehrtagestouren als eine Gesamt-GPX ("combined") oder als eine GPX je Tagesetappe ("per_day")
    gpx_mode: Mapped[str] = mapped_column(String(20), default="combined", server_default="combined")
    role: Mapped[str] = mapped_column(String(20), default=UserRole.user)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
