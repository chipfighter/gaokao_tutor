"""Supervisor node — LLM-based intent classification and routing."""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import TutorState
from src.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT

_VALID_INTENTS = {"academic", "planning", "emotional"}


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.0,
    )


def supervisor_node(state: TutorState) -> dict:
    """Classify the latest user message into an intent category."""
    llm = _get_llm()

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ])

    try:
        parsed = json.loads(response.content.strip())
        intent = parsed.get("intent", "academic")
    except (json.JSONDecodeError, AttributeError):
        intent = "academic"

    if intent not in _VALID_INTENTS:
        intent = "academic"

    return {"intent": intent}


def route_by_intent(state: TutorState) -> str:
    """Conditional edge function: route to the appropriate subgraph."""
    return state.get("intent", "academic")
