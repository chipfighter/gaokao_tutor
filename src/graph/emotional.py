"""Emotional response node — single LLM call with homeroom teacher persona."""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import TutorState
from src.prompts.emotional import EMOTIONAL_SYSTEM_PROMPT


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.8,
    )


def emotional_response(state: TutorState) -> dict:
    """Respond with warm, practical emotional support."""
    llm = _get_llm()

    history = [SystemMessage(content=EMOTIONAL_SYSTEM_PROMPT)]
    for msg in state["messages"]:
        history.append(msg)

    response = llm.invoke(history)

    return {"messages": [AIMessage(content=response.content)]}
