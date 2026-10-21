import re
import unicodedata


def export_filename(title: str, extension: str) -> str:
    """ASCII-Dateiname aus dem Abenteuertitel, z. B. "Radtour Ulm → Lindau" → "radtour-ulm-lindau.gpx"."""
    ascii_title = unicodedata.normalize("NFKD", title.replace("ß", "ss")).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_title).strip("-").lower()[:80]
    return f"{slug or 'abenteuer'}.{extension}"
