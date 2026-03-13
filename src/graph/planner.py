"""SubGraph B — Study Planner: draft plan, policy search, refinement."""

from __future__ import annotations

import os
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import TutorState
from src.prompts.planner import (
    PLANNER_INIT_PROMPT,
    PLANNER_REFINE_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from src.tools.search_tool import search_tool


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )


# ── Node 1: generate initial plan ───────────────────────────────────

def init_plan(state: TutorState) -> dict:
    """Create a draft study plan from the user's request."""
    llm = _get_llm()

    last_msg = state["messages"][-1]
    user_request = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    prompt = PLANNER_INIT_PROMPT.format(user_request=user_request)
    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"plan": response.content}


# ── Node 2: search latest Gaokao policies ───────────────────────────

def search_policy(state: TutorState) -> dict:
    """Use Tavily to fetch the latest Gaokao policy information."""
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

    try:
        results = search_tool.invoke(query)
        if isinstance(results, str):
            search_results = [{"content": results, "title": "", "url": ""}]
        else:
            search_results = [
                {
                    "content": r.get("content", ""),
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                }
                for r in results
            ]
    except Exception:
        search_results = []

    return {"search_results": search_results}


# ── Node 3: refine plan with policy info ─────────────────────────────

def refine_plan(state: TutorState) -> dict:
    """Merge draft plan with policy search results into final output."""
    llm = _get_llm()

    draft = state.get("plan", "")
    search_results = state.get("search_results", [])

    policy_info = "\n\n".join(
        f"- {r.get('title', '无标题')}: {r.get('content', '')}"
        for r in search_results
    ) if search_results else "未获取到最新政策信息，请基于通用经验给出建议。"

    prompt = PLANNER_REFINE_PROMPT.format(
        draft_plan=draft,
        policy_info=policy_info,
    )

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"messages": [AIMessage(content=response.content)]}
