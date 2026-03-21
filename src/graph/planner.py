"""SubGraph B — Study Planner: policy search, then single-call plan generation.

Notes:
- 2-step (search → generate)

Future Roadmap:
- [Local RAG] Replace/Augment web search with a VectorDB (ChromaDB) containing
  official PDF documents from provincial education examination authorities.
- [Context Filtering] Implement a re-ranking stage to prioritize official
  .gov.cn domains over social media/marketing content.
- [Provincial Routing] Automatically inject user's provincial context into
  search queries to handle diverse Gaokao schemas (e.g., 3+1+2 vs. 3+3).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.llm import get_fallback_llm, invoke_with_fallback
from src.graph.state import TutorState
from src.prompts.planner import PLANNER_GENERATE_PROMPT, PLANNER_SYSTEM_PROMPT
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_search


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )


# ── Node 1: search latest Gaokao policies ─────────────────────────

# Time Limit to prevent search too long
_SEARCH_TIMEOUT = 15


@traced_node
def search_policy(state: TutorState) -> dict:
    """Use DuckDuckGo to fetch the latest Gaokao policy information. Times out after 15s."""
    year = datetime.now().year
    query = f"{year}年高考最新政策 考试时间安排 科目改革"

    with traced_search(query=query, timeout=_SEARCH_TIMEOUT) as span:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(web_search_fn, query)
                search_results = future.result(timeout=_SEARCH_TIMEOUT)
            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute("search.timed_out", False)
        except TimeoutError:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", True)
        except Exception:
            search_results = []
            span.set_attribute("search.result_count", 0)
            span.set_attribute("search.timed_out", False)

    return {"search_results": search_results}


# ── Node 2: generate complete plan ────────────────────────────────

@traced_node
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

    fallback = get_fallback_llm(temperature=0.7)
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_plan",
        temperature=0.7,
    ) as span:
        response = invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}
