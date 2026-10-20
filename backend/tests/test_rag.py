import httpx

from app.db import SessionLocal
from app.rag.ingest import ingest_document, split_markdown
from app.rag.retrieve import search_chunks

DOC = b"""# Testdokument

## Abschnitt A

Erster Absatz.

## Abschnitt B

### Unterpunkt

Zweiter Absatz.
"""


def test_split_markdown_keeps_heading_path() -> None:
    chunks = split_markdown(DOC.decode())
    assert [c.heading for c in chunks] == [
        "Testdokument > Abschnitt A",
        "Testdokument > Abschnitt B > Unterpunkt",
    ]


def test_split_markdown_respects_max_chars() -> None:
    text = "# T\n\n" + "\n\n".join("x" * 400 for _ in range(10))
    assert all(len(c.text) <= 1000 for c in split_markdown(text, max_chars=1000))


async def test_ingest_replace_and_search() -> None:
    async with SessionLocal() as session:
        await ingest_document(session, "test.md", DOC)
        document = await ingest_document(session, "test.md", DOC)  # ersetzt die erste Version
        assert document.title == "Testdokument"
        results = await search_chunks(session, "Zweiter Absatz", min_score=-1)
    assert len(results) == 2
    assert all(r.document == "Testdokument" for r in results)


async def test_document_upload_admin_only(
    client: httpx.AsyncClient, admin_headers: dict[str, str], user_headers: dict[str, str]
) -> None:
    files = {"file": ("tipps.md", DOC, "text/markdown")}
    assert (await client.post("/api/documents", headers=user_headers, files=files)).status_code == 403

    response = await client.post("/api/documents", headers=admin_headers, files=files)
    assert response.status_code == 201
    assert response.json()["chunk_count"] == 2

    bad = await client.post("/api/documents", headers=admin_headers, files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400

    doc_id = response.json()["id"]
    assert (await client.delete(f"/api/documents/{doc_id}", headers=admin_headers)).status_code == 204
    assert (await client.get("/api/documents", headers=admin_headers)).json() == []
