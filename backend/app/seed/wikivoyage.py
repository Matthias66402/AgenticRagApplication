"""Importiert Wikivoyage-Artikel in die Wissensbasis.

Beispiele:
    python -m app.seed.wikivoyage "Bodensee-Radweg" "Europäischer Fernwanderweg E5"
    python -m app.seed.wikivoyage --no-ingest "Donauradweg"       # nur als Datei speichern, später: python -m app.seed
    python -m app.seed.wikivoyage --dry-run "Allgäu"               # Markdown nur anzeigen
    python -m app.seed.wikivoyage --search "Radweg Bayern"         # passende Artikeltitel finden

Gespeichert wird unter app/seed/documents/wikivoyage-<titel>.md (versionierbar); ein erneuter Import
desselben Artikels ersetzt Datei und Wissensbasis-Eintrag.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from app.config import get_settings
from app.db import SessionLocal, engine
from app.rag.ingest import ingest_document
from app.rag.wikivoyage import ArticleNotFound, fetch_article, search_titles

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.seed.wikivoyage", description=__doc__.split("\n")[0])
    parser.add_argument("titles", nargs="+", help="Artikeltitel (oder Suchbegriffe mit --search)")
    parser.add_argument("--lang", default="de", help="Sprachversion von Wikivoyage (Standard: de)")
    parser.add_argument("--no-ingest", action="store_true", help="nur als Datei speichern, nicht einlesen")
    parser.add_argument("--dry-run", action="store_true", help="Markdown nur ausgeben, nichts speichern")
    parser.add_argument("--search", action="store_true", help="nur nach passenden Artikeltiteln suchen")
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    try:
        return await run(parse_args(argv))
    finally:
        await engine.dispose()


async def run(args: argparse.Namespace) -> int:
    failures = 0
    headers = {"User-Agent": get_settings().nominatim_user_agent}
    async with httpx.AsyncClient(headers=headers, timeout=30) as http:
        if args.search:
            for query in args.titles:
                print(f"„{query}“: " + ", ".join(f"„{t}“" for t in await search_titles(http, query, args.lang, 10)))
            return 0

        for title in args.titles:
            try:
                article = await fetch_article(http, title, args.lang)
            except ArticleNotFound as exc:
                failures += 1
                hint = ", ".join(f"„{s}“" for s in exc.suggestions) or "keine Treffer"
                print(f"✗ „{title}“ nicht gefunden. Meintest Du: {hint}")
                continue
            except httpx.HTTPError as exc:
                failures += 1
                print(f"✗ „{title}“: Wikivoyage nicht erreichbar ({exc})")
                continue

            if args.dry_run:
                print(article.markdown)
                continue

            path = DOCUMENTS_DIR / article.filename
            path.write_text(article.markdown, encoding="utf-8")
            status = f"✓ „{article.title}“ → {path.name} ({len(article.markdown) // 1024} KB, Stand {article.timestamp[:10]})"
            if args.no_ingest:
                print(status + ", nicht eingelesen")
                continue
            try:
                async with SessionLocal() as session:
                    document = await ingest_document(session, path.name, path.read_bytes())
                print(status + f", {len(document.chunks)} Abschnitte eingelesen")
            except Exception as exc:  # z. B. Embedding-Anbieter gesperrt
                failures += 1
                print(status + f", Einlesen fehlgeschlagen: {exc}\n  Datei ist gespeichert; später mit `python -m app.seed` einlesen.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
