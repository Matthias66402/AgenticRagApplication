"""Wikivoyage-Artikel abrufen und als Markdown mit Quellen- und Lizenzangabe aufbereiten.

Inhalte von Wikivoyage stehen unter CC BY-SA 4.0: Weitergabe nur mit Namensnennung (Link auf Artikel
und Versionsgeschichte) und unter gleicher Lizenz.
"""

import re
import unicodedata
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

API_URL = "https://{lang}.wikivoyage.org/w/api.php"
LICENSE_NAME = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.de"
LICENSE = f"{LICENSE_NAME} ({LICENSE_URL})"

# Kopfzeile, die fetch_article jedem Artikel voranstellt; parse_attribution liest sie wieder aus
ATTRIBUTION_RE = re.compile(
    r"^> Quelle: (?P<publisher>Wikivoyage), Artikel „(?P<title>.+?)“ \((?P<url>\S+)\), "
    r"Version vom (?P<revision_date>\d{4}-\d{2}-\d{2})\. Autoren: siehe (?P<history_url>\S+)\. "
    r"Lizenz: (?P<license>.+?) \((?P<license_url>\S+)\)\.$",
    re.MULTILINE,
)

# Abschnitte ohne Nutzen für die Tourenplanung (Linklisten, Literaturangaben)
SKIP_SECTIONS = {"literatur", "weblinks", "einzelnachweise", "anmerkungen", "quellen", "further reading", "external links"}

SECTION_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


class Attribution(BaseModel):
    """Namensnennung nach CC BY-SA: Titel, Link, Autoren (Versionsgeschichte), Lizenz mit Link."""

    publisher: str
    title: str
    url: str
    revision_date: str
    history_url: str
    license: str
    license_url: str


def parse_attribution(text: str) -> Attribution | None:
    match = ATTRIBUTION_RE.search(text)
    return Attribution(**match.groupdict()) if match else None


class ArticleNotFound(Exception):
    def __init__(self, title: str, suggestions: list[str]):
        super().__init__(title)
        self.title = title
        self.suggestions = suggestions


@dataclass
class WikivoyageArticle:
    title: str
    url: str
    revision_id: int
    timestamp: str
    markdown: str

    @property
    def filename(self) -> str:
        ascii_title = unicodedata.normalize("NFKD", self.title.replace("ß", "ss")).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_title).strip("-").lower()
        return f"wikivoyage-{slug}.md"


def extract_to_markdown(extract: str) -> str:
    """Wandelt den Klartext-Auszug (`== Abschnitt ==`) in Markdown um; leere und irrelevante Abschnitte entfallen."""
    sections: list[tuple[int, str, list[str]]] = [(1, "", [])]
    for line in extract.splitlines():
        if match := SECTION_RE.match(line):
            sections.append((len(match.group(1)), match.group(2), []))
        else:
            sections[-1][2].append(line.rstrip())

    parts: list[str] = []
    skip_level: int | None = None
    for level, heading, lines in sections:
        if skip_level is not None and level > skip_level:
            continue  # Unterabschnitt eines übersprungenen Abschnitts
        skip_level = None
        if heading.lower() in SKIP_SECTIONS:
            skip_level = level
            continue
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if heading:
            # Leere Überschrift nur behalten, wenn Unterabschnitte mit Inhalt folgen (wird unten bereinigt)
            parts.append(f"{'#' * level} {heading}")
        if body:
            parts.append(body)

    # Überschriften ohne Inhalt bis zur nächsten gleich- oder höherrangigen Überschrift entfernen
    cleaned: list[str] = []
    for i, part in enumerate(parts):
        if part.startswith("#"):
            level = len(part) - len(part.lstrip("#"))
            following = parts[i + 1] if i + 1 < len(parts) else None
            if following is None or (
                following.startswith("#") and len(following) - len(following.lstrip("#")) <= level
            ):
                continue
        cleaned.append(part)
    return "\n\n".join(cleaned)


async def search_titles(http: httpx.AsyncClient, query: str, lang: str = "de", limit: int = 5) -> list[str]:
    response = await http.get(
        API_URL.format(lang=lang),
        params={"action": "query", "format": "json", "formatversion": 2, "list": "search", "srsearch": query, "srlimit": limit},
    )
    response.raise_for_status()
    return [hit["title"] for hit in response.json()["query"]["search"]]


async def fetch_article(http: httpx.AsyncClient, title: str, lang: str = "de") -> WikivoyageArticle:
    """Lädt einen Artikel (Weiterleitungen werden aufgelöst). Wirft ArticleNotFound mit Titelvorschlägen."""
    response = await http.get(
        API_URL.format(lang=lang),
        params={
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
            "prop": "extracts|info|revisions", "explaintext": 1, "exsectionformat": "wiki",
            "inprop": "url", "rvprop": "timestamp|ids",
        },
    )  # fmt: skip
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise ArticleNotFound(title, await search_titles(http, title, lang))

    revision = page["revisions"][0]
    body = extract_to_markdown(page["extract"])
    history_url = f"{page['fullurl']}?action=history"
    header = (
        f"# {page['title']}\n\n"
        f"> Quelle: Wikivoyage, Artikel „{page['title']}“ ({page['fullurl']}), "
        f"Version vom {revision['timestamp'][:10]}. Autoren: siehe {history_url}. Lizenz: {LICENSE}."
    )
    return WikivoyageArticle(
        title=page["title"],
        url=page["fullurl"],
        revision_id=revision["revid"],
        timestamp=revision["timestamp"],
        markdown=f"{header}\n\n{body}\n",
    )
