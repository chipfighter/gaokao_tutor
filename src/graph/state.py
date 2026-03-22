"""TutorState: the shared state object that flows through all nodes in the LangGraph, acting as the single source of truth for the system."""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]                 # Chat history
    intent: Literal["academic", "planning", "emotional"]    # User intent
    subject: str                                            # The topic being discussed
    keypoints: list[str]                                    # Key points
    context: Annotated[list[dict], operator.add]            # Merged retrieval context (fan-in)
    search_results: list[dict]                              # Planner search results
    plan: str                                               # Generated plans
    retry_count: int                                        # Hallucination retry counter
    hallucination_detected: bool                            # Hallucination flag
