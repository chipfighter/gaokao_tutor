"""TutorState — single source of truth flowing through the entire LangGraph."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Literal["academic", "planning", "emotional"]
    subject: str
    keypoints: list[str]
    retrieved_docs: list[dict]
    search_results: list[dict]
    plan: str
