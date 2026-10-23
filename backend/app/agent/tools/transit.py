import datetime as dt
import json
import re
from typing import Any
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

import httpx
from pydantic_ai import RunContext

from app.agent.deps import AgentDeps
from app.agent.tools.common import http_error_message, tool_error

MAX_WALK_SECONDS = 30 * 60  # Fußweg zur ersten bzw. von der letzten Haltestelle
MIN_WALK_MINUTES = 3  # kürzere Fußwege innerhalb einer Station (Bahnsteigwechsel) nicht ausgeben
# Bahn und Nahverkehr für die letzte Meile; ohne Fernbusse (COACH) und Flüge
TRANSIT_MODES = "RAIL,SUBURBAN,METRO,SUBWAY,TRAM,BUS,FERRY,FUNICULAR,AERIAL_LIFT,CABLE_CAR"
MAX_CONNECTIONS = 3
# Bus, Tram usw. nur für den Weg zum bzw. vom Bahnhof: Varianten mit denselben Zügen gelten als eine Verbindung
LOCAL_MODES = {"WALK", "BUS", "TRAM", "SUBWAY", "FERRY", "FUNICULAR", "AERIAL_LIFT", "AREAL_LIFT", "CABLE_CAR"}
TRACK_PREFIX = re.compile(r"^(Gl(eis)?|Bstg)\.?\s*", re.IGNORECASE)  # manche Feeds liefern "Gl4" statt "4"
ATTRIBUTION = "Fahrplandaten: Transitous (transitous.org)"
# Liegt die nächste Verbindung weiter von der Wunschzeit entfernt, fehlen nach dem Fahrplanwechsel vermutlich Daten
MAX_GAP_HOURS = 3
BAHN_SEARCH_URL = "https://www.bahn.de/buchung/fahrplan/suche"
# Nahverkehrsfeeds liefern oft nur die Nummer ("24"); ohne Verkehrsmittel ist unklar, ob Bus oder Straßenbahn
MODE_LABELS = {
    "BUS": "Bus", "TRAM": "Tram", "SUBWAY": "U-Bahn", "FERRY": "Fähre", "FUNICULAR": "Standseilbahn",
    "AERIAL_LIFT": "Seilbahn", "AREAL_LIFT": "Seilbahn", "CABLE_CAR": "Seilbahn",
}  # fmt: skip


def timetable_change(year: int) -> dt.date:
    """Fahrplanwechsel in Europa: in der Nacht zum zweiten Sonntag im Dezember."""
    first = dt.date(year, 12, 1)
    return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7)


def next_timetable_change(today: dt.date) -> dt.date:
    change = timetable_change(today.year)
    return change if today < change else timetable_change(today.year + 1)


def bahn_search_url(
    from_name: str, from_lat: float, from_lon: float, to_name: str, to_lat: float, to_lon: float,
    moment: dt.datetime, arrive_by: bool,
) -> str:
    """Vorausgefüllte Verbindungssuche auf bahn.de; Orte als Koordinaten, damit kein Bahnhofs-ID nötig ist."""

    def location(name: str, lat: float, lon: float) -> str:
        return f"A=2@O={name}@X={round(lon * 1_000_000)}@Y={round(lat * 1_000_000)}@"

    params = {
        "sts": "true", "so": from_name, "zo": to_name,
        "soid": location(from_name, from_lat, from_lon), "zoid": location(to_name, to_lat, to_lon),
        "hd": f"{moment:%Y-%m-%dT%H:%M:%S}", "hza": "A" if arrive_by else "D", "kl": "2",
    }  # fmt: skip
    return f"{BAHN_SEARCH_URL}#{urlencode(params, quote_via=quote)}"


def _gap_hours(connections: list[dict[str, Any]], moment: dt.datetime, arrive_by: bool) -> float:
    """Abstand der zeitlich nächsten Verbindung zur Wunschzeit (Abfahrt bzw. bei arrive_by Ankunft)."""
    wanted = moment.replace(tzinfo=None)
    return min(
        abs((dt.datetime.fromisoformat(c["arrival" if arrive_by else "departure"]) - wanted).total_seconds()) / 3600
        for c in connections
    )


def _local(timestamp: str, tz: str | None) -> str:
    """UTC-Zeitstempel der API → Ortszeit der Haltestelle, z. B. 2026-10-24T07:36."""
    moment = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return moment.astimezone(ZoneInfo(tz or "Europe/Berlin")).strftime("%Y-%m-%dT%H:%M")


def _stop_name(place: dict[str, Any], fallback: str) -> str:
    # Bei Koordinaten als Start/Ziel heißen die Endpunkte START bzw. END
    return fallback if place.get("name") in ("START", "END", None) else place["name"]


def _same_place(a: str, b: str) -> bool:
    # "Oldenburg (Oldb) Hbf" vs. "Oldenburg(Oldb) Hbf"
    return a.replace(" ", "").casefold() == b.replace(" ", "").casefold()


def _line(leg: dict[str, Any]) -> str:
    """Linienbezeichnung, bei reinen Nummern mit Verkehrsmittel: "Bus 24", "Tram 1"; "ICE 595", "U3" bleiben."""
    name = (leg.get("displayName") or leg.get("routeShortName") or "").strip()
    label = MODE_LABELS.get(leg["mode"])
    if not name:
        return label or leg["mode"]
    return f"{label} {name}" if label and name[0].isdigit() else name


def _leg(leg: dict[str, Any], from_name: str, to_name: str) -> dict[str, Any] | None:
    origin, destination = leg["from"], leg["to"]
    result: dict[str, Any] = {
        "line": _line(leg),
        "from_stop": _stop_name(origin, from_name),
        "to_stop": _stop_name(destination, to_name),
        "departure": _local(leg["startTime"], origin.get("tz")),
        "arrival": _local(leg["endTime"], destination.get("tz")),
    }
    if leg["mode"] == "WALK":
        minutes = round(leg["duration"] / 60)
        if minutes < MIN_WALK_MINUTES and _same_place(result["from_stop"], result["to_stop"]):
            return None
        return result | {"line": f"Fußweg ({minutes} min)"}
    if leg.get("headsign"):
        result["direction"] = leg["headsign"]
    if origin.get("track"):
        result["track"] = TRACK_PREFIX.sub("", origin["track"])
    if leg.get("cancelled"):
        result["cancelled"] = True
    return result


def _distinct(itineraries: list[dict[str, Any]], arrive_by: bool) -> list[dict[str, Any]]:
    """Je Zugfolge nur die Variante mit den wenigsten Umstiegen bzw. der kürzesten Dauer; davon die Verbindungen,
    die der gewünschten Zeit am nächsten liegen (bei `arrive_by` die spätesten Ankünfte, sonst die frühesten
    Abfahrten)."""
    best: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for itinerary in sorted(itineraries, key=lambda it: (it["transfers"], it["duration"])):
        # Zug = Linie + Abfahrtszeit, sonst fielen z. B. alle RS30-Fahrten eines Morgens zusammen
        trains = tuple(
            (leg.get("displayName") or leg.get("routeShortName") or "", leg["startTime"])
            for leg in itinerary["legs"]
            if leg["mode"] not in LOCAL_MODES
        )
        best.setdefault(trains, itinerary)
    if arrive_by:
        closest = sorted(best.values(), key=lambda it: it["endTime"], reverse=True)
    else:
        closest = sorted(best.values(), key=lambda it: it["startTime"])
    return closest[:MAX_CONNECTIONS]


def _connection(itinerary: dict[str, Any], from_name: str, to_name: str) -> dict[str, Any]:
    legs = itinerary["legs"]
    return {
        "departure": _local(itinerary["startTime"], legs[0]["from"].get("tz")),
        "arrival": _local(itinerary["endTime"], legs[-1]["to"].get("tz")),
        "duration_minutes": round(itinerary["duration"] / 60),
        "transfers": itinerary["transfers"],
        "legs": [leg for raw in legs if (leg := _leg(raw, from_name, to_name))],
    }


async def find_train_connections(
    ctx: RunContext[AgentDeps],
    from_name: str,
    from_lat: float,
    from_lon: float,
    to_name: str,
    to_lat: float,
    to_lon: float,
    date: dt.date,
    time: dt.time = dt.time(8, 0),
    arrive_by: bool = False,
    timezone: str = "Europe/Berlin",
) -> dict[str, Any]:
    """Sucht Bahnverbindungen (inkl. Bus/Tram für die letzte Meile) für An- und Abreise (Transitous-Fahrplandaten).

    Für die Anreise vom Wohnort zum Tourstart `arrive_by=True` mit der gewünschten Ankunftszeit nutzen,
    für die Rückreise vom Tourziel die früheste Abfahrtszeit nach Etappenende. Koordinaten per `geocode`;
    liegt der Ort weit ab von Haltestellen, den nächsten Talort oder Bahnhof verwenden.
    Liefert keine Preise und keine verlässliche Angabe zur Fahrradmitnahme.

    Args:
        from_name: Kurzer Name des Abfahrtsorts, z. B. "Berlin".
        from_lat: Breitengrad des Abfahrtsorts.
        from_lon: Längengrad des Abfahrtsorts.
        to_name: Kurzer Name des Zielorts, z. B. "Garmisch-Partenkirchen".
        to_lat: Breitengrad des Zielorts.
        to_lon: Längengrad des Zielorts.
        date: Reisedatum (ISO).
        time: Früheste Abfahrt bzw. bei arrive_by späteste Ankunft (Ortszeit, HH:MM).
        arrive_by: True = Ankunft spätestens zu `time`, False = Abfahrt frühestens zu `time`.
        timezone: IANA-Zeitzone für `time`, Standard Europe/Berlin.

    Returns:
        Bis zu 3 Verbindungen mit Abfahrt, Ankunft (Ortszeit), Dauer, Umstiegen und Teilstrecken
        (Linie, Richtung, Gleis).
    """
    settings = ctx.deps.settings
    key = (round(from_lat, 4), round(from_lon, 4), round(to_lat, 4), round(to_lon, 4), date, time, arrive_by)
    if key in ctx.deps.train_queries:
        # Wiederholte identische Abfragen kosten nur LLM-Anfragen; das Modell soll mit dem Ergebnis weiterarbeiten
        return ctx.deps.train_queries[key]["result"] | {
            "hint": "Diese Abfrage wurde bereits ausgeführt, das Ergebnis ist unverändert. Nicht erneut abfragen, "
            "sondern mit dem Ergebnis weiterarbeiten bzw. antworten."
        }
    if not settings.transit_api_url:
        return tool_error("Fahrplanauskunft ist nicht konfiguriert. An- und Abreise ohne konkrete Verbindungen planen.")
    if date < ctx.deps.today:
        return tool_error(f"{date} liegt in der Vergangenheit (heute ist {ctx.deps.today}).")
    try:
        moment = dt.datetime.combine(date, time, tzinfo=ZoneInfo(timezone))
    except Exception:
        return tool_error(f"Unbekannte Zeitzone '{timezone}'.")

    params = {
        "fromPlace": f"{from_lat},{from_lon}",
        "toPlace": f"{to_lat},{to_lon}",
        "time": moment.isoformat(),
        "arriveBy": str(arrive_by).lower(),
        "transitModes": TRANSIT_MODES,
        "numItineraries": 2 * MAX_CONNECTIONS,  # Spielraum für das Zusammenfassen gleicher Zugfolgen
        "maxPreTransitTime": MAX_WALK_SECONDS,
        "maxPostTransitTime": MAX_WALK_SECONDS,
        "detailedTransfers": "false",
    }
    try:
        response = await ctx.deps.http.get(
            settings.transit_api_url, params=params, headers={"User-Agent": settings.nominatim_user_agent}
        )
        response.raise_for_status()
        itineraries = response.json().get("itineraries", [])
    except httpx.HTTPError as exc:
        return tool_error(http_error_message("Transitous", exc))

    connections = [_connection(it, from_name, to_name) for it in _distinct(itineraries, arrive_by)]
    connections.sort(key=lambda c: c["departure"])
    bahn_url = bahn_search_url(from_name, from_lat, from_lon, to_name, to_lat, to_lon, moment, arrive_by)
    change = next_timetable_change(ctx.deps.today)
    if date >= change and (not connections or _gap_hours(connections, moment, arrive_by) > MAX_GAP_HOURS):
        # Transitous lädt den neuen Jahresfahrplan der DB erst um den Fahrplanwechsel; bis dahin fehlen die Züge
        result = {
            "connections": [],
            "error": f"Für {date:%d.%m.%Y} (nach dem Fahrplanwechsel am {change:%d.%m.%Y}) liegen noch keine "
            "Bahnfahrpläne vor. Keine Verbindung angeben oder erfinden. Ein Link zur Verbindungssuche bei bahn.de "
            "erscheint automatisch unter der Antwort bzw. im Plan unter An- und Abreise; nur kurz darauf hinweisen, "
            "die URL nicht in `sources` oder Notizen kopieren. Gehört die Fahrt zu einer Tour, gib den Plan trotzdem "
            "mit `final_plan` aus (ohne diese Verbindung).",
            "bahn_url": bahn_url,
        }
    elif not connections:
        return tool_error(
            f"Keine Verbindung {from_name} → {to_name} gefunden. Nächstgelegenen Bahnhof oder andere Uhrzeit "
            f"versuchen oder auf die Verbindungssuche bei bahn.de verweisen: {bahn_url}"
        )
    else:
        result = {
            "connections": connections,
            "note": "Fahrplan ohne Gewähr, ohne Preise und Echtzeitdaten. Vor Reiseantritt bei der Bahn prüfen.",
            "source": ATTRIBUTION,
        }
        if date >= change:
            result |= {
                "note": f"Nach dem Fahrplanwechsel am {change:%d.%m.%Y}: Fahrplandaten können noch unvollständig "
                "sein. Ein Link zum Prüfen bei bahn.de erscheint automatisch unter der Antwort bzw. im Plan.",
                "bahn_url": bahn_url,
            }
    ctx.deps.train_queries[key] = {
        "from_name": from_name, "to_name": to_name, "date": date, "time": time, "arrive_by": arrive_by, "result": result,
    }  # fmt: skip
    return result


WEEKDAYS = ["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."]


def _duration(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d} h"


def _query_label(query: dict[str, Any]) -> str:
    """z. B. "Oldenburg → Kiel · So. 21.02.2027, ab 06:00"."""
    date: dt.date = query["date"]
    mode = "an" if query["arrive_by"] else "ab"
    return (
        f"{query['from_name']} → {query['to_name']} · {WEEKDAYS[date.weekday()]} {date:%d.%m.%Y}, "
        f"{mode} {query['time']:%H:%M}"
    )


def timetable_links(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Links zur Verbindungssuche bei bahn.de für Abfragen nach dem Fahrplanwechsel (fehlende bzw. unsichere Daten);
    das Backend hängt sie an den Plan, damit das LLM keine URLs abschreiben muss."""
    return [
        {"label": _query_label(query), "url": url, "missing": not query["result"]["connections"]}
        for query in queries
        if (url := query["result"].get("bahn_url"))
    ]


def history_timetable_links(
    history: list[dict[str, Any]], first_day: dt.date, last_day: dt.date
) -> list[dict[str, Any]]:
    """bahn.de-Links aller Bahnabfragen im Verlauf (auch früherer Nachrichten) mit Reisedatum im Zeitraum; der Agent
    fragt die Bahn oft in einer Nachricht ab und gibt den Plan erst in einer späteren aus. Gleiche Strecke und Tag:
    die neueste Abfrage."""
    calls: dict[str, dict[str, Any]] = {}
    links: dict[tuple[str, str, str], dict[str, Any]] = {}
    for message in history:
        for part in message.get("parts", []):
            if part.get("tool_name") != "find_train_connections":
                continue
            if part.get("part_kind") == "tool-call":
                args = part.get("args")
                calls[part.get("tool_call_id", "")] = json.loads(args) if isinstance(args, str) else args or {}
            elif part.get("part_kind") == "tool-return" and isinstance(result := part.get("content"), dict):
                args = calls.get(part.get("tool_call_id", ""), {})
                try:
                    query = {
                        "from_name": args["from_name"], "to_name": args["to_name"],
                        "date": dt.date.fromisoformat(args["date"]),
                        "time": dt.time.fromisoformat(args.get("time", "08:00")),
                        "arrive_by": args.get("arrive_by", False), "result": result,
                    }  # fmt: skip
                except (KeyError, TypeError, ValueError):
                    continue
                if first_day <= query["date"] <= last_day and (found := timetable_links([query])):
                    key = (query["from_name"], query["to_name"], args["date"])
                    links.pop(key, None)  # neueste Abfrage ans Ende
                    links[key] = found[0]
    return list(links.values())


def format_train_queries(queries: list[dict[str, Any]]) -> str:
    """Verbindungen als lesbarer Text für die Chat-Antwort; das LLM muss sie so nicht selbst wiedergeben."""
    blocks = []
    for query in queries:
        lines = [f"🚆 {_query_label(query)}"]
        for number, connection in enumerate(query["result"]["connections"], start=1):
            transfers = connection["transfers"]
            lines.append(
                f"\n{number}) {connection['departure'][11:]} → {connection['arrival'][11:]} · "
                f"{_duration(connection['duration_minutes'])} · "
                + ("ohne Umstieg" if not transfers else f"{transfers} {'Umstieg' if transfers == 1 else 'Umstiege'}")
            )
            for leg in connection["legs"]:
                if leg["line"].startswith("Fußweg"):
                    lines.append(f"   🚶 {leg['line']} → {leg['to_stop']}")
                    continue
                track = f" (Gl. {leg['track']})" if leg.get("track") else ""
                cancelled = " – fällt aus!" if leg.get("cancelled") else ""
                lines.append(
                    f"   {leg['departure'][11:]} {leg['line']}: {leg['from_stop']}{track} → "
                    f"{leg['to_stop']} {leg['arrival'][11:]}{cancelled}"
                )
        if bahn_url := query["result"].get("bahn_url"):
            if query["result"]["connections"]:
                lines.append(f"\nNach dem Fahrplanwechsel evtl. unvollständig: [bei bahn.de prüfen]({bahn_url})")
            else:
                lines.append(f"\nFür diesen Tag noch keine Fahrpläne: [Verbindungen bei bahn.de]({bahn_url})")
        blocks.append("\n".join(lines))
    blocks.append(f"{ATTRIBUTION} · ohne Gewähr, ohne Echtzeit und Preise.")
    return "\n\n".join(blocks)
