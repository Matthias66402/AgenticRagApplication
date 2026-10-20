import math
from typing import Any

import httpx


def tool_error(message: str) -> dict[str, Any]:
    """Fehler als Tool-Ergebnis: Das LLM kann darauf reagieren (anderer Ort, Schätzung, Rückfrage)."""
    return {"error": message}


def http_error_message(service: str, exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"{service} hat nicht rechtzeitig geantwortet. Später erneut versuchen oder ohne diese Daten planen."
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{service} meldet HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    return f"{service} ist nicht erreichbar: {exc}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
