import asyncio
import datetime as dt
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import http_error_message, tool_error

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 7  # danach sind Tagesvorhersagen kaum besser als Klimawerte
CLIMATE_YEARS = 3

WMO_CODES = {
    0: "klar", 1: "überwiegend klar", 2: "teilweise bewölkt", 3: "bedeckt",
    45: "Nebel", 48: "Reifnebel",
    51: "leichter Niesel", 53: "Niesel", 55: "starker Niesel", 56: "gefrierender Niesel", 57: "gefrierender Niesel",
    61: "leichter Regen", 63: "Regen", 65: "starker Regen", 66: "gefrierender Regen", 67: "gefrierender Regen",
    71: "leichter Schneefall", 73: "Schneefall", 75: "starker Schneefall", 77: "Schneegriesel",
    80: "Regenschauer", 81: "Regenschauer", 82: "heftige Regenschauer",
    85: "Schneeschauer", 86: "starke Schneeschauer",
    95: "Gewitter", 96: "Gewitter mit Hagel", 99: "Gewitter mit Hagel",
}  # fmt: skip


async def _fetch(http: httpx.AsyncClient, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = await http.get(url, params=params)
    response.raise_for_status()
    return response.json()["daily"]


def _shift(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # 29. Februar
        return d.replace(year=d.year - years, day=28)


async def _forecast(
    http: httpx.AsyncClient, base: dict[str, Any], start: dt.date, end: dt.date
) -> list[dict[str, Any]]:
    daily = await _fetch(http, FORECAST_URL, base | {
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,wind_speed_10m_max",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    })  # fmt: skip
    return [
        {
            "date": daily["time"][i],
            "kind": "forecast",
            "conditions": WMO_CODES.get(daily["weather_code"][i], "unbekannt"),
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "precipitation_probability_pct": daily["precipitation_probability_max"][i],
            "wind_max_kmh": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]


async def _climate(
    http: httpx.AsyncClient, base: dict[str, Any], start: dt.date, end: dt.date, today: dt.date
) -> list[dict[str, Any]]:
    """Mittel derselben Kalendertage der letzten Jahre (nur abgeschlossene Jahre im Archiv)."""
    offset = start.year - today.year + 1
    years = await asyncio.gather(*[
        _fetch(http, ARCHIVE_URL, base | {
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "start_date": _shift(start, offset + n).isoformat(),
            "end_date": _shift(end, offset + n).isoformat(),
        })
        for n in range(CLIMATE_YEARS)
    ])  # fmt: skip

    def mean(key: str, i: int) -> float | None:
        values = [y[key][i] for y in years if i < len(y.get(key, [])) and y[key][i] is not None]
        return round(sum(values) / len(values), 1) if values else None

    return [
        {
            "date": (start + dt.timedelta(days=i)).isoformat(),
            "kind": "climate",
            "temp_max_c": mean("temperature_2m_max", i),
            "temp_min_c": mean("temperature_2m_min", i),
            "precipitation_mm": mean("precipitation_sum", i),
        }
        for i in range((end - start).days + 1)
    ]


async def get_weather(
    ctx: RunContext[AgentDeps], lat: float, lon: float, start_date: dt.date, days: int = 1
) -> dict[str, Any]:
    """Liefert das Wetter für einen Ort und Zeitraum (Open-Meteo).

    Tage innerhalb der nächsten Woche bekommen eine Vorhersage. Weiter in der Zukunft ist eine Vorhersage nicht
    belastbar; dafür kommen Klimawerte (Mittel der letzten Jahre am selben Datum), also typisches Wetter.

    Args:
        lat: Breitengrad.
        lon: Längengrad.
        start_date: Erster Tag (ISO-Datum).
        days: Anzahl Tage (1-14).
    """
    days = max(1, min(days, 14))
    today = ctx.deps.today
    if start_date < today:
        return tool_error(f"{start_date} liegt in der Vergangenheit (heute ist {today}).")
    end_date = start_date + dt.timedelta(days=days - 1)
    horizon = today + dt.timedelta(days=FORECAST_HORIZON_DAYS - 1)
    base = {"latitude": lat, "longitude": lon, "timezone": "auto"}

    try:
        result_days: list[dict[str, Any]] = []
        if start_date <= horizon:
            result_days += await _forecast(ctx.deps.http, base, start_date, min(end_date, horizon))
        if end_date > horizon:
            climate_start = max(start_date, horizon + dt.timedelta(days=1))
            result_days += await _climate(ctx.deps.http, base, climate_start, end_date, today)
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Open-Meteo", exc))

    kinds = {d["kind"] for d in result_days}
    if kinds == {"forecast"}:
        return {"kind": "forecast", "days": result_days}
    climate_note = (
        f"Für Tage nach {horizon} ist keine verlässliche Vorhersage möglich; dort stehen Mittelwerte der letzten "
        f"{CLIMATE_YEARS} Jahre am selben Datum. Beschreibe sie im Plan als typisches Wetter der Jahreszeit, nicht "
        "als Vorhersage, und rate, kurz vor der Tour die aktuelle Vorhersage zu prüfen."
    )
    return {"kind": "climate" if kinds == {"climate"} else "mixed", "note": climate_note, "days": result_days}
