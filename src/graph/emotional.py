"""Emotional response node — single LLM call with homeroom teacher persona."""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import get_setting, load_prompt
from src.graph.llm import get_fallback_llm, invoke_with_fallback
from src.graph.state import TutorState
from src.tracing import traced_llm_call, traced_node


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=get_setting("emotional.temperature", 0.8),
    )


@traced_node
def emotional_response(state: TutorState) -> dict:
    """Respond with warm, practical emotional support."""
    llm = _get_llm()

    history = [SystemMessage(content=load_prompt("emotional_system"))]
    for msg in state["messages"]:
        history.append(msg)

    temperature = get_setting("emotional.temperature", 0.8)
    fallback = get_fallback_llm(temperature=temperature)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="emotional_response",
        temperature=temperature,
    ) as span:
        response = invoke_with_fallback(
            llm, history, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}
