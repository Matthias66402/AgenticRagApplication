from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.rag.embeddings import embed_query


class RetrievedChunk(BaseModel):
    document: str
    heading: str
    text: str
    score: float


async def search_chunks(
    session: AsyncSession, query: str, top_k: int = 4, min_score: float = 0.25
) -> list[RetrievedChunk]:
    """Kosinus-Ähnlichkeitssuche über pgvector (HNSW-Index)."""
    vector = await embed_query(query)
    distance = Chunk.embedding.cosine_distance(vector)
    rows = await session.execute(
        select(Chunk, Document.title, distance.label("distance"))
        .join(Document)
        .order_by(distance)
        .limit(top_k)
    )
    results = [
        RetrievedChunk(
            document=title,
            heading=chunk.meta.get("heading", ""),
            text=chunk.text,
            score=round(1 - dist, 3),
        )
        for chunk, title, dist in rows
    ]
    return [r for r in results if r.score >= min_score]
