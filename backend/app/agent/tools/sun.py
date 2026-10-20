import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import sun


def sun_times(lat: float, lon: float, date: dt.date, timezone: str = "Europe/Berlin") -> dict[str, Any]:
    """Berechnet Sonnenaufgang, Sonnenuntergang und Tageslichtdauer für einen Ort und Tag.

    Hilfreich, um zu prüfen, ob eine Etappe bei Tageslicht machbar ist.

    Args:
        lat: Breitengrad.
        lon: Längengrad.
        date: Datum (ISO).
        timezone: IANA-Zeitzone, Standard Europe/Berlin (gilt für DE, AT, CH, IT).
    """
    try:
        tz = ZoneInfo(timezone)
        times = sun(Observer(latitude=lat, longitude=lon), date=date, tzinfo=tz)
    except Exception as exc:  # unbekannte Zeitzone, Polartag/-nacht
        return {"error": f"Sonnenzeiten nicht berechenbar: {exc}"}
    daylight = times["sunset"] - times["sunrise"]
    return {
        "sunrise": times["sunrise"].strftime("%H:%M"),
        "sunset": times["sunset"].strftime("%H:%M"),
        "civil_dusk": times["dusk"].strftime("%H:%M"),
        "daylight_hours": round(daylight.total_seconds() / 3600, 1),
    }
