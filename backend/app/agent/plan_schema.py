"""Strukturierte Ausgaben des Planungs-Agenten."""

import datetime as dt
from enum import StrEnum
from typing import Any, Self

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


# Fahrplanzeiten als Ortszeit-Text wie im Tool-Ergebnis. Ein datetime-Feld verlangt im strict-Modus eine Zeitzone,
# dabei würde das Modell Ortszeiten womöglich in UTC umrechnen.
LOCAL_TIME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$"


def local_time(description: str = "Ortszeit YYYY-MM-DDTHH:MM, unverändert aus find_train_connections") -> Any:
    return Field(pattern=LOCAL_TIME_PATTERN, description=description)


class JourneyLeg(BaseModel):
    line: str = Field(description='Linie wie "ICE 595" oder "Fußweg (8 min)"')
    from_stop: str
    to_stop: str
    departure: str = local_time()
    arrival: str = local_time()
    direction: str | None = None
    track: str | None = None


class Journey(BaseModel):
    """Bahnverbindung für An- oder Abreise; nur aus find_train_connections-Ergebnissen übernehmen."""

    departure: str = local_time()
    arrival: str = local_time()
    duration_minutes: int = Field(ge=0)
    transfers: int = Field(ge=0)
    legs: list[JourneyLeg] = Field(min_length=1)
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
    outbound_journey: Journey | None = Field(
        default=None, description="Anreise mit Bahn/ÖPNV zum Tourstart, falls gewünscht"
    )
    return_journey: Journey | None = Field(default=None, description="Rückreise mit Bahn/ÖPNV vom Tourziel")
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


class TextAnswer(BaseModel):
    """Antwort ohne Abenteuerplan, z. B. eine reine Bahnverbindungsauskunft."""

    message: str = Field(
        description="Kurze Antwort. Gefundene Bahnverbindungen werden automatisch darunter angezeigt, nicht wiederholen."
    )


class ClarificationQuestion(BaseModel):
    """Rückfrage an den Nutzer, wenn Pflichtangaben fehlen oder unklar sind."""

    question: str = Field(description="Freundliche Rückfrage, die alle fehlenden Punkte auf einmal abdeckt")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="z. B. transport, start_date, duration_days, start, destination, accommodation_type",
    )
