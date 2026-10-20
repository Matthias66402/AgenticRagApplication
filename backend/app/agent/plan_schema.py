"""Strukturierte Ausgaben des Planungs-Agenten."""

import datetime as dt
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class TransportMode(StrEnum):
    hiking = "hiking"
    cycling = "cycling"
    mtb = "mtb"
    car = "car"
    canoe = "canoe"
    public_transport = "public_transport"


class AccommodationType(StrEnum):
    hut = "hut"
    camping = "camping"
    hotel = "hotel"
    hostel = "hostel"
    bivouac = "bivouac"


# Obergrenzen pro Tagesetappe für die Plausibilitätsprüfung (km)
MAX_STAGE_KM: dict[TransportMode, float] = {
    TransportMode.hiking: 35,
    TransportMode.cycling: 160,
    TransportMode.mtb: 90,
    TransportMode.canoe: 45,
    TransportMode.car: 900,
    TransportMode.public_transport: 1500,
}


class Place(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class AccommodationStop(BaseModel):
    name: str
    type: AccommodationType
    lat: float
    lon: float
    website: str | None = None


class Stage(BaseModel):
    day: int = Field(ge=1, description="Tag der Tour, beginnend bei 1")
    date: dt.date
    start: Place
    end: Place
    distance_km: float = Field(ge=0)
    ascent_m: float | None = Field(default=None, ge=0)
    descent_m: float | None = Field(default=None, ge=0)
    duration_hours: float | None = Field(default=None, ge=0, description="Reine Bewegungszeit")
    accommodation: AccommodationStop | None = Field(
        default=None, description="Übernachtung am Etappenende; nur aus find_accommodation-Ergebnissen"
    )
    notes: str | None = None


class AdventurePlan(BaseModel):
    """Vollständiger Abenteuerplan. Nur ausgeben, wenn alle Pflichtangaben bekannt sind."""

    title: str = Field(max_length=200)
    summary: str = Field(description="2-4 Sätze Überblick über die Tour")
    transport: TransportMode
    start_date: dt.date
    duration_days: int = Field(ge=1, le=30)
    start: Place
    destination: Place
    accommodation_type: AccommodationType | None = Field(
        default=None, description="Pflicht bei mehrtägigen Touren (duration_days > 1)"
    )
    stages: list[Stage] = Field(min_length=1, description="Genau eine Etappe pro Tag")
    route_ids: list[str] = Field(
        default_factory=list, description="route_id-Werte aus plan_route, die die Gesamtroute bilden"
    )
    weather_summary: str | None = None
    packing_tips: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(
        default_factory=list, description="Genutzte Wissensbasis-Dokumente und Datendienste"
    )

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.duration_days > 1 and self.accommodation_type is None:
            raise ValueError("Bei mehrtägigen Touren ist accommodation_type Pflicht – frage den Nutzer danach")
        if len(self.stages) != self.duration_days:
            raise ValueError(
                f"Es müssen genau {self.duration_days} Etappen sein (eine pro Tag), nicht {len(self.stages)}"
            )
        return self


class ClarificationQuestion(BaseModel):
    """Rückfrage an den Nutzer, wenn Pflichtangaben fehlen oder unklar sind."""

    question: str = Field(description="Freundliche Rückfrage, die alle fehlenden Punkte auf einmal abdeckt")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="z. B. transport, start_date, duration_days, start, destination, accommodation_type",
    )
