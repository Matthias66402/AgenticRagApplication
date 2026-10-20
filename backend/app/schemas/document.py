from datetime import datetime

from pydantic import BaseModel


class DocumentRead(BaseModel):
    id: int
    title: str
    source: str
    chunk_count: int
    created_at: datetime
