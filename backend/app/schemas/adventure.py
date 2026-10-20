from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import AdventureStatus


class AdventureCreate(BaseModel):
    title: str = Field(default="Neues Abenteuer", min_length=1, max_length=200)


class AdventureUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class AdventureSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    status: AdventureStatus
    created_at: datetime
    updated_at: datetime


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    tool_calls: list[Any]
    created_at: datetime


class AdventureRead(AdventureSummary):
    plan: dict[str, Any] | None
    messages: list[ChatMessageRead]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
