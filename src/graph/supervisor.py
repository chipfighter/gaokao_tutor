"""Supervisor node — LLM-based intent classification, subject detection, and keypoint extraction.

Combines routing and academic keypoint extraction into a single LLM call
to eliminate a redundant API roundtrip on the academic path.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_setting, load_prompt
from src.graph.llm import get_node_llm
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

_VALID_INTENTS = set(get_setting("supervisor.valid_intents", ["academic", "planning", "emotional"]))


@traced_node
async def supervisor_node(state: TutorState) -> dict:
    """Classify intent, detect subject, and extract keypoints in one LLM call.

    Returns:
        Dict with ``intent``, ``subject``, and ``keypoints`` for state update.
    """
    llm = get_node_llm("supervisor")

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    temperature = get_setting("supervisor.temperature", 0.0)
    model_name = get_setting("supervisor.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    with traced_llm_call(
        model_name=model_name,
        node_name="supervisor",
        temperature=temperature,
    ):
        response = await llm.ainvoke([
            SystemMessage(content=load_prompt("supervisor_system")),
            HumanMessage(content=user_text),
        ])

    try:
        parsed = json.loads(response.content.strip())
        intent = parsed.get("intent", "academic")
        subject = parsed.get("subject", "other")
        keypoints = parsed.get("keypoints", [])
    except (json.JSONDecodeError, AttributeError):
        intent = "academic"
        subject = "other"
        keypoints = []

    if intent not in _VALID_INTENTS:
        intent = "academic"

    return {"intent": intent, "subject": subject, "keypoints": keypoints}


def route_by_intent(state: TutorState) -> str:
    """Conditional edge function: route to the appropriate subgraph."""
    return state.get("intent", "academic")
