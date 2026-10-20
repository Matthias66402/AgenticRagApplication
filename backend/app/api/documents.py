from fastapi import APIRouter, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.auth.deps import AdminUser, SessionDep
from app.models import Chunk, Document
from app.rag.ingest import SUPPORTED_SUFFIXES, ingest_document
from app.schemas.document import DocumentRead

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def document_rows(session: SessionDep, document_id: int | None = None) -> list[DocumentRead]:
    query = (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk)
        .group_by(Document.id)
        .order_by(Document.title)
    )
    if document_id is not None:
        query = query.where(Document.id == document_id)
    return [
        DocumentRead(id=d.id, title=d.title, source=d.source, chunk_count=count, created_at=d.created_at)
        for d, count in await session.execute(query)
    ]


@router.get("", response_model=list[DocumentRead])
async def list_documents(_: AdminUser, session: SessionDep) -> list[DocumentRead]:
    return await document_rows(session)


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, admin: AdminUser, session: SessionDep) -> DocumentRead:
    filename = file.filename or ""
    if not filename.lower().endswith(SUPPORTED_SUFFIXES):
        raise HTTPException(status_code=400, detail=f"Erlaubte Formate: {', '.join(SUPPORTED_SUFFIXES)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Datei ist größer als 10 MB")
    try:
        document = await ingest_document(session, filename, data, uploaded_by=admin.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await document_rows(session, document.id))[0]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: int, _: AdminUser, session: SessionDep) -> None:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    await session.delete(document)
    await session.commit()
