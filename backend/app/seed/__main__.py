"""Legt den Admin-Nutzer an, liest die Beispieldokumente in die Wissensbasis ein und ergänzt fehlende
Quellenangaben in bestehenden Plänen.

Aufruf: ``python -m app.seed`` (idempotent; ``--force`` liest vorhandene Dokumente neu ein).
"""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

from app.auth.security import hash_password
from app.config import get_settings
from app.db import SessionLocal, engine
from app.models import Adventure, Document, User, UserRole
from app.rag.attribution import knowledge_attributions
from app.rag.ingest import ingest_document

DOCUMENTS_DIR = Path(__file__).parent / "documents"


async def seed_admin() -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        if await session.scalar(select(User).where(User.email == settings.admin_email.lower())):
            print(f"Admin {settings.admin_email} existiert bereits")
            return
        session.add(
            User(
                email=settings.admin_email.lower(),
                display_name="Administrator",
                password_hash=hash_password(settings.admin_password),
                role=UserRole.admin,
            )
        )
        await session.commit()
        print(f"Admin {settings.admin_email} angelegt")


async def seed_documents(force: bool = False) -> None:
    async with SessionLocal() as session:
        existing = set(await session.scalars(select(Document.source)))
        for path in sorted(DOCUMENTS_DIR.glob("*.md")):
            if path.name in existing and not force:
                print(f"  übersprungen: {path.name}")
                continue
            document = await ingest_document(session, path.name, path.read_bytes())
            print(f"  eingelesen: {path.name} ({len(document.chunks)} Abschnitte)")


async def backfill_attributions() -> None:
    """Pläne aus der Zeit vor der Quellenangabe: genutzte Wikivoyage-Artikel aus dem Agent-Verlauf ergänzen."""
    async with SessionLocal() as session:
        adventures = await session.scalars(select(Adventure).where(Adventure.plan.is_not(None)))
        updated = 0
        for adventure in adventures:
            if "attributions" in adventure.plan:
                continue
            attributions = await knowledge_attributions(session, adventure.agent_history or [])
            adventure.plan = adventure.plan | {"attributions": attributions}
            updated += 1
        await session.commit()
        print(f"Quellenangaben ergänzt: {updated} Pläne")


async def main() -> None:
    await seed_admin()
    print("Wissensbasis:")
    await seed_documents(force="--force" in sys.argv)
    await backfill_attributions()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
