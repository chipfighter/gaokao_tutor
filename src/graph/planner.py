"""SubGraph B — Study Planner: policy search, then single-call plan generation.

Optimized from 3-step (init → search → refine) to 2-step (search → generate)
to eliminate one LLM roundtrip.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import TutorState
from src.prompts.planner import PLANNER_GENERATE_PROMPT, PLANNER_SYSTEM_PROMPT
from src.tools.search_tool import search as web_search_fn


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )


# ── Node 1: search latest Gaokao policies ─────────────────────────

_SEARCH_TIMEOUT = 15


def search_policy(state: TutorState) -> dict:
    """Use DuckDuckGo to fetch the latest Gaokao policy information. Times out after 15s."""
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(web_search_fn, query)
            search_results = future.result(timeout=_SEARCH_TIMEOUT)
    except (TimeoutError, Exception):
        search_results = []

    return {"search_results": search_results}


# ── Node 2: generate complete plan ────────────────────────────────

def generate_plan(state: TutorState) -> dict:
    """Produce a complete study plan from user request + policy context in one LLM call."""
    llm = _get_llm()

    last_msg = state["messages"][-1]
    user_request = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    search_results = state.get("search_results", [])
    policy_info = "\n\n".join(
        f"- {r.get('title', '无标题')}: {r.get('content', '')}"
        for r in search_results
    ) if search_results else "未获取到最新政策信息，请基于通用经验给出建议。"

    prompt = PLANNER_GENERATE_PROMPT.format(
        user_request=user_request,
        policy_info=policy_info,
    )

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"messages": [AIMessage(content=response.content)]}
