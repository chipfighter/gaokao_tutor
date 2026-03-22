"""Supervisor node — LLM-based intent classification, subject detection, and keypoint extraction.

Combines routing and academic keypoint extraction into a single LLM call
to eliminate a redundant API roundtrip on the academic path.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_setting, load_prompt
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node

_VALID_INTENTS = set(get_setting("supervisor.valid_intents", ["academic", "planning", "emotional"]))


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=get_setting("supervisor.temperature", 0.0),
    )


@traced_node
def supervisor_node(state: TutorState) -> dict:
    """Classify intent, detect subject, and extract keypoints in one LLM call.

    Returns:
        Dict with ``intent``, ``subject``, and ``keypoints`` for state update.
    """
    llm = _get_llm()

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    temperature = get_setting("supervisor.temperature", 0.0)
    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="supervisor",
        temperature=temperature,
    ):
        response = llm.invoke([
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
