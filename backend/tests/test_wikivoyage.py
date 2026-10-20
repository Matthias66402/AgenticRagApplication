import httpx
import pytest
import respx

from app.rag.wikivoyage import API_URL, ArticleNotFound, extract_to_markdown, fetch_article
from app.seed import wikivoyage as cli

EXTRACT = """Einleitung zum Radweg.


== Streckenprofil ==

Länge: 200 km

== Vorbereitung ==

=== Etappen ===

Eine Woche einplanen.

=== Leer ===

== Literatur ==

Ein Buch.

=== Karten ===

Eine Karte.

== Sicherheit ==

Helm tragen."""

PAGE = {
    "query": {
        "pages": [
            {
                "title": "Testradweg",
                "fullurl": "https://de.wikivoyage.org/wiki/Testradweg",
                "revisions": [{"revid": 42, "timestamp": "2026-10-03T11:15:26Z"}],
                "extract": EXTRACT,
            }
        ]
    }
}
URL = API_URL.format(lang="de")


def test_extract_to_markdown() -> None:
    markdown = extract_to_markdown(EXTRACT)
    assert markdown == (
        "Einleitung zum Radweg.\n\n"
        "## Streckenprofil\n\nLänge: 200 km\n\n"
        "## Vorbereitung\n\n### Etappen\n\nEine Woche einplanen.\n\n"
        "## Sicherheit\n\nHelm tragen."
    )


@respx.mock
async def test_fetch_article_adds_source_and_license() -> None:
    respx.get(URL).respond(json=PAGE)
    async with httpx.AsyncClient() as http:
        article = await fetch_article(http, "Testradweg")
    assert article.filename == "wikivoyage-testradweg.md"
    assert article.markdown.startswith("# Testradweg\n\n> Quelle: Wikivoyage")
    assert "Version vom 2026-10-03" in article.markdown
    assert "?action=history" in article.markdown and "CC BY-SA 4.0" in article.markdown


@respx.mock
async def test_missing_article_suggests_titles() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": "Europäischer Fernwanderweg E5"}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": "E5", "missing": True}]}})

    respx.get(URL).mock(side_effect=respond)
    async with httpx.AsyncClient() as http:
        with pytest.raises(ArticleNotFound) as exc_info:
            await fetch_article(http, "E5")
    assert exc_info.value.suggestions == ["Europäischer Fernwanderweg E5"]


@respx.mock
async def test_cli_saves_file_and_ingests(tmp_path, monkeypatch, client, admin_headers, capsys) -> None:
    monkeypatch.setattr(cli, "DOCUMENTS_DIR", tmp_path)
    respx.get(URL).respond(json=PAGE)
    respx.route(host="test").pass_through()  # ASGI-Testclient nicht abfangen

    assert await cli.run(cli.parse_args(["Testradweg"])) == 0

    saved = tmp_path / "wikivoyage-testradweg.md"
    assert saved.read_text(encoding="utf-8").startswith("# Testradweg")
    assert "Abschnitte eingelesen" in capsys.readouterr().out
    documents = (await client.get("/api/documents", headers=admin_headers)).json()
    assert [(d["title"], d["source"]) for d in documents] == [("Testradweg", "wikivoyage-testradweg.md")]


@respx.mock
async def test_cli_no_ingest_only_saves(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "DOCUMENTS_DIR", tmp_path)
    respx.get(URL).respond(json=PAGE)
    assert await cli.run(cli.parse_args(["--no-ingest", "Testradweg"])) == 0
    assert (tmp_path / "wikivoyage-testradweg.md").exists()
    assert "nicht eingelesen" in capsys.readouterr().out
