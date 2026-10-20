import os

# Vor jedem App-Import auf die Test-Datenbank umstellen
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://adventure:adventure@localhost:5440/adventure_test"
)

from collections.abc import AsyncIterator, Iterator  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from pydantic_ai.embeddings import TestEmbeddingModel  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.auth.security import hash_password  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User, UserRole  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402

PASSWORD = "secret-pass-123"


@pytest.fixture(autouse=True)
async def clean_db() -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE users, adventures, chat_messages, documents, chunks RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture(scope="session", autouse=True)
async def dispose_engine() -> AsyncIterator[None]:
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
def fake_embeddings() -> Iterator[None]:
    with get_embedder().override(model=TestEmbeddingModel(dimensions=get_settings().embedding_dim)):
        yield


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def create_user(email: str, role: UserRole = UserRole.user, name: str = "Test") -> User:
    async with SessionLocal() as session:
        user = User(email=email, display_name=name, password_hash=hash_password(PASSWORD), role=role)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def auth_headers(client: httpx.AsyncClient, email: str) -> dict[str, str]:
    response = await client.post("/api/auth/login", data={"username": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def user_headers(client: httpx.AsyncClient) -> dict[str, str]:
    await create_user("user@example.com", name="Ute")
    return await auth_headers(client, "user@example.com")


@pytest.fixture
async def admin_headers(client: httpx.AsyncClient) -> dict[str, str]:
    await create_user("admin@example.com", role=UserRole.admin, name="Admin")
    return await auth_headers(client, "admin@example.com")
