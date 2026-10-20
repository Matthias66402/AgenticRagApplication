import asyncio
import datetime as dt
from typing import Any

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import http_error_message, tool_error

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15
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


async def get_weather(
    ctx: RunContext[AgentDeps], lat: float, lon: float, start_date: dt.date, days: int = 1
) -> dict[str, Any]:
    """Liefert das Wetter für einen Ort und Zeitraum (Open-Meteo).

    Liegt der Zeitraum innerhalb der nächsten ~2 Wochen, kommt eine Vorhersage.
    Weiter in der Zukunft werden Klimawerte (Mittel der letzten Jahre am gleichen Datum) geliefert.

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
    base = {"latitude": lat, "longitude": lon, "timezone": "auto"}

    try:
        if start_date <= today + dt.timedelta(days=FORECAST_HORIZON_DAYS):
            forecast_end = min(end_date, today + dt.timedelta(days=FORECAST_HORIZON_DAYS))
            daily = await _fetch(ctx.deps.http, FORECAST_URL, base | {
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max",
                "start_date": start_date.isoformat(),
                "end_date": forecast_end.isoformat(),
            })  # fmt: skip
            result = {
                "kind": "forecast",
                "days": [
                    {
                        "date": daily["time"][i],
                        "conditions": WMO_CODES.get(daily["weather_code"][i], "unbekannt"),
                        "temp_max_c": daily["temperature_2m_max"][i],
                        "temp_min_c": daily["temperature_2m_min"][i],
                        "precipitation_mm": daily["precipitation_sum"][i],
                        "precipitation_probability_pct": daily["precipitation_probability_max"][i],
                        "wind_max_kmh": daily["wind_speed_10m_max"][i],
                    }
                    for i in range(len(daily["time"]))
                ],
            }
            if forecast_end < end_date:
                result["note"] = f"Vorhersage reicht nur bis {forecast_end}; spätere Tage sind noch unsicher."
            return result

        # Klimawerte: dieselben Kalendertage der letzten Jahre mitteln
        def shift(d: dt.date, years: int) -> dt.date:
            try:
                return d.replace(year=d.year - years)
            except ValueError:  # 29. Februar
                return d.replace(year=d.year - years, day=28)

        offset = start_date.year - today.year + 1
        years = await asyncio.gather(*[
            _fetch(ctx.deps.http, ARCHIVE_URL, base | {
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "start_date": shift(start_date, offset + n).isoformat(),
                "end_date": shift(end_date, offset + n).isoformat(),
            })
            for n in range(CLIMATE_YEARS)
        ])  # fmt: skip
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Open-Meteo", exc))

    def mean(key: str, i: int) -> float | None:
        values = [y[key][i] for y in years if i < len(y[key]) and y[key][i] is not None]
        return round(sum(values) / len(values), 1) if values else None

    return {
        "kind": "climate",
        "note": f"Keine Vorhersage möglich; Mittelwerte der letzten {CLIMATE_YEARS} Jahre am selben Datum.",
        "days": [
            {
                "date": (start_date + dt.timedelta(days=i)).isoformat(),
                "temp_max_c": mean("temperature_2m_max", i),
                "temp_min_c": mean("temperature_2m_min", i),
                "precipitation_mm": mean("precipitation_sum", i),
            }
            for i in range(days)
        ],
    }
