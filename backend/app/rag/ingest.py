"""Dokumente laden → in Abschnitte zerlegen → einbetten → in pgvector speichern."""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.rag.embeddings import embed_documents

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")
MAX_CHUNK_CHARS = 1200
EMBED_BATCH_SIZE = 64

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class TextChunk:
    text: str
    heading: str


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def split_markdown(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[TextChunk]:
    """Teilt an Überschriften, danach an Absätzen. Jeder Chunk trägt seinen Überschriftenpfad als Kontext."""
    sections: list[tuple[str, list[str]]] = []
    heading_path: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in current):
            sections.append((" > ".join(heading_path), current.copy()))
        current.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_path[:] = heading_path[: level - 1] + [match.group(2).strip()]
        else:
            current.append(line)
    flush()

    chunks: list[TextChunk] = []
    for heading, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) + 2 > max_chars:
                chunks.append(TextChunk(buffer, heading))
                buffer = ""
            # Überlange Absätze hart teilen
            while len(paragraph) > max_chars:
                chunks.append(TextChunk(paragraph[:max_chars], heading))
                paragraph = paragraph[max_chars:]
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if buffer:
            chunks.append(TextChunk(buffer, heading))
    return chunks


def guess_title(source: str, text: str) -> str:
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            return match.group(2).strip()
    return source.rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()


async def ingest_document(
    session: AsyncSession, source: str, data: bytes, uploaded_by: int | None = None
) -> Document:
    """Liest ein Dokument ein. Eine vorhandene Version mit gleichem `source` wird ersetzt."""
    if not source.lower().endswith(SUPPORTED_SUFFIXES):
        raise ValueError(f"Nicht unterstütztes Format. Erlaubt: {', '.join(SUPPORTED_SUFFIXES)}")

    text = extract_text(source, data)
    title = guess_title(source, text)
    pieces = split_markdown(text)
    if not pieces:
        raise ValueError("Das Dokument enthält keinen Text")

    vectors: list[list[float]] = []
    for start in range(0, len(pieces), EMBED_BATCH_SIZE):
        batch = pieces[start : start + EMBED_BATCH_SIZE]
        # Überschrift mit einbetten, damit kurze Abschnitte ihren Kontext behalten
        vectors.extend(await embed_documents([f"{title} – {p.heading}\n{p.text}" for p in batch]))

    existing = await session.scalar(select(Document).where(Document.source == source))
    if existing is not None:
        await session.delete(existing)
        await session.flush()

    document = Document(title=title, source=source, uploaded_by=uploaded_by)
    document.chunks = [
        Chunk(position=i, text=piece.text, meta={"heading": piece.heading}, embedding=vector)
        for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
    ]
    session.add(document)
    await session.commit()
    return document
