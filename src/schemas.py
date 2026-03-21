"""Data structures that can be reused across modules."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str
    thread_id: str | None = None
