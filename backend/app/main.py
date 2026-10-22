from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import adventures, auth, chat, documents, users
from app.config import get_settings
from app.observability import setup_tracing, shutdown_tracing

# Vor dem ersten Agent-Lauf: Pydantic AI instrumentieren und Langfuse-Export einrichten
setup_tracing()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    shutdown_tracing()


app = FastAPI(
    title="AgenticRAG Abenteuerplaner",
    description="Plant Abenteuer mit einem Agenten, der RAG und externe Tools dynamisch kombiniert.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (auth.router, users.router, adventures.router, chat.router, documents.router):
    app.include_router(router)


@app.get("/api/health", tags=["meta"])
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "llm_model": settings.llm_model, "embedding_model": settings.embedding_model}
