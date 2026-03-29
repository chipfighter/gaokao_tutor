"""TutorState: the shared state object that flows through all nodes in the LangGraph, acting as the single source of truth for the system."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Sentinel value: returning this from a node signals "clear all context"
CONTEXT_CLEAR: list[dict] = [{"__clear__": True}]


def context_reducer(existing: list[dict], update: list[dict]) -> list[dict]:
    """Merge context lists from fan-out branches.

    Returning CONTEXT_CLEAR resets context to empty (used on retry path).
    Normal updates are appended (same as operator.add).
    """
    if update and update[0].get("__clear__"):
        return []
    return existing + update


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]                 # Chat history
    intent: Literal["academic", "planning", "emotional"]    # User intent
    subject: str                                            # The topic being discussed
    keypoints: list[str]                                    # Key points
    context: Annotated[list[dict], context_reducer]         # Merged retrieval context (fan-in)
    search_results: list[dict]                              # Planner search results
    plan: str                                               # Generated plans
    retry_count: int                                        # Hallucination retry counter
    hallucination_detected: bool                            # Hallucination flag
    rewritten_query: str                                    # Rewritten query on retry
    hallucination_reason: str                               # Reason from hallucination eval
