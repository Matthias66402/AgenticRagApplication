from functools import lru_cache

from pydantic_ai import Embedder

from app.config import get_settings


@lru_cache
def get_embedder() -> Embedder:
    """Anbieter-agnostischer Embedder; das Modell kommt aus EMBEDDING_MODEL (z. B. openai:text-embedding-3-small).

    In Tests: ``with get_embedder().override(model=TestEmbeddingModel(dimensions=...))``.
    """
    return Embedder(get_settings().embedding_model)


def _check_dimensions(vectors: list[list[float]]) -> list[list[float]]:
    expected = get_settings().embedding_dim
    for vector in vectors:
        if len(vector) != expected:
            raise ValueError(
                f"Embedding hat {len(vector)} Dimensionen, erwartet {expected}. "
                "EMBEDDING_DIM anpassen, Migration erzeugen und Dokumente neu einlesen."
            )
    return vectors


async def embed_documents(texts: list[str]) -> list[list[float]]:
    result = await get_embedder().embed_documents(texts)
    return _check_dimensions([list(v) for v in result.embeddings])


async def embed_query(text: str) -> list[float]:
    result = await get_embedder().embed_query(text)
    return _check_dimensions([list(result.embeddings[0])])[0]
