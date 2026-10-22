"""Tracing mit Langfuse.

Pydantic AI erzeugt OpenTelemetry-Spans für Agent-Läufe, Modellaufrufe (als Generation mit Modell, Tokens, Kosten),
Tool-Aufrufe und Embeddings; das Langfuse-SDK exportiert sie. Je Chat-Nachricht entsteht ein Trace mit der Wurzel
`chat-turn` (Eingabe: Nutzernachricht, Ausgabe: Antwort), je Abenteuer eine Session.

Aktiv nur mit LANGFUSE_PUBLIC_KEY und LANGFUSE_SECRET_KEY (Basis-URL: LANGFUSE_BASE_URL, Umgebung:
LANGFUSE_TRACING_ENVIRONMENT); LANGFUSE_TRACING_ENABLED=false schaltet es ab, z. B. in den Tests.
"""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_ai import Agent, Embedder

from app.config import get_settings

logger = logging.getLogger(__name__)

ADDRESS_PLACEHOLDER = "[Adresse]"
STREET_PLACEHOLDER = "[Straße]"
# "Nelkenstraße 10a" → Straßenname "Nelkenstraße" (für Geocoding-Ergebnisse wie "10, Nelkenstraße, Oldenburg")
HOUSE_NUMBER = re.compile(r"\s+\d+\s*[a-zA-Z]?(\s*[-/]\s*\d+\s*[a-zA-Z]?)?$")

_enabled = False
# Wohnadressen aus Profilen, die vor dem Export ersetzt werden; wird beim Chat registriert (siehe register_home_address)
_addresses: set[str] = set()
_streets: set[str] = set()
_address_pattern: re.Pattern[str] | None = None
_street_pattern: re.Pattern[str] | None = None


def tracing_enabled() -> bool:
    return _enabled


def setup_tracing() -> None:
    """Einmal beim Start: Langfuse-Client anlegen und Pydantic AI instrumentieren."""
    global _enabled
    settings = get_settings()
    if not (settings.langfuse_tracing_enabled and settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    from langfuse import Langfuse

    # Zugangsdaten, Basis-URL und Umgebung liest das SDK aus den LANGFUSE_*-Umgebungsvariablen
    Langfuse(mask_otel_spans=mask_home_addresses)
    Agent.instrument_all()
    Embedder.instrument_all()
    _enabled = True
    logger.info("Langfuse-Tracing aktiv")


def shutdown_tracing() -> None:
    """Gepufferte Spans beim Beenden senden."""
    if _enabled:
        from langfuse import get_client

        get_client().shutdown()


def register_home_address(address: str | None) -> None:
    """Merkt sich Straße und Hausnummer eines Nutzers, damit sie in keinem exportierten Span landen."""
    global _address_pattern, _street_pattern
    address = (address or "").strip()
    if not _enabled or not address or address in _addresses:
        return
    _addresses.add(address)
    if street := HOUSE_NUMBER.sub("", address).strip():
        _streets.add(street)
    # Längere zuerst, damit "Nelkenstraße 10" vor "Nelkenstraße" greift; Muster werden als Ganzes ersetzt (threadsicher)
    _address_pattern = _compile(_addresses)
    _street_pattern = _compile(_streets) if _streets else None


def _compile(values: set[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(v) for v in sorted(values, key=len, reverse=True)), re.IGNORECASE)


def mask_text(text: str) -> str:
    if _address_pattern:
        text = _address_pattern.sub(ADDRESS_PLACEHOLDER, text)
    if _street_pattern:
        text = _street_pattern.sub(STREET_PLACEHOLDER, text)
    return text


def mask_home_addresses(*, params: Any) -> Any:
    """`mask_otel_spans`-Hook: ersetzt registrierte Wohnadressen in allen Text-Attributen eines Export-Batches."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    if not _address_pattern:
        return None
    patches = {}
    for identifier, span in params.spans.items():
        changed: dict[str, Any] = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                if (masked := mask_text(value)) != value:
                    changed[key] = masked
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
                if (masked_list := [mask_text(v) for v in value]) != list(value):
                    changed[key] = masked_list
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


class _NoopObservation:
    def update(self, **_: Any) -> None:
        pass


@contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Eigene Observation (z. B. `retriever`) im aktuellen Trace; ohne Tracing ein Platzhalter mit `update()`."""
    if not _enabled:
        yield _NoopObservation()
        return
    from langfuse import get_client

    with get_client().start_as_current_observation(name=name, as_type=as_type, **kwargs) as observation:
        yield observation


@contextmanager
def chat_turn(*, message: str, user_id: int, adventure_id: int) -> Iterator[Any]:
    """Wurzel eines Traces je Chat-Nachricht; User und Session gelten für alle Observations darin."""
    if not _enabled:
        yield _NoopObservation()
        return
    from langfuse import propagate_attributes

    with observe("chat-turn", as_type="span", input=message) as root, propagate_attributes(
        user_id=str(user_id),
        session_id=f"adventure-{adventure_id}",
        trace_name="chat-turn",
        tags=["chat"],
        metadata={"adventure_id": str(adventure_id)},
    ):
        yield root
