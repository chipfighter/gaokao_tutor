"""SubGraph A — Academic Tutor: RAG retrieval, web search fallback, answer generation.

Keypoint extraction is handled by the supervisor node (merged for latency),
so this subgraph starts directly at RAG retrieval.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.llm import get_fallback_llm, invoke_with_fallback
from src.graph.state import TutorState
from src.prompts.academic import ACADEMIC_ANSWER_PROMPT, ACADEMIC_SYSTEM_PROMPT
from src.rag.retriever import RELEVANCE_THRESHOLD, retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search


def _get_llm(**kwargs) -> ChatOpenAI:
    defaults = dict(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )
    defaults.update(kwargs)
    return ChatOpenAI(**defaults)


# ── Node 1: RAG retrieval ─────────────────────────────────────────

@traced_node
def rag_retrieve(state: TutorState) -> dict:
    """Query ChromaDB with keypoints extracted by the supervisor node."""
    keypoints = state.get("keypoints", [])
    subject = state.get("subject")
    query = " ".join(keypoints) if keypoints else state["messages"][-1].content

    with traced_retrieval(query=query, subject=subject if subject != "other" else None) as span:
        result = retrieve(query=query, subject=subject if subject != "other" else None)
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    return {"retrieved_docs": result["docs"]}


def should_web_search(state: TutorState) -> str:
    """Conditional edge: route to web_search if RAG missed, else generate_answer."""
    docs = state.get("retrieved_docs", [])
    if not docs or docs[0].get("score", 0) < RELEVANCE_THRESHOLD:
        return "web_search"
    return "generate_answer"


# ── Node 2: web search (conditional) ──────────────────────────────

_SEARCH_TIMEOUT = 15


@traced_node
def web_search(state: TutorState) -> dict:
    """Fallback to DuckDuckGo when RAG retrieval misses. Times out after 15s."""
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

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


# ── Node 3: generate answer ──────────────────────────────────────

def _format_retrieved(docs: list[dict]) -> str:
    if not docs:
        return "无相关参考资料。"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{d.get('source', '未知')}（相关度：{d.get('score', 'N/A')}）\n{d.get('content', '')}")
    return "\n\n".join(parts)


def _format_search(results: list[dict]) -> str:
    if not results:
        return "无网络搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title', '无标题')} ({r.get('url', '')})\n{r.get('content', '')}")
    return "\n\n".join(parts)


@traced_node
def generate_answer(state: TutorState) -> dict:
    """Synthesize final answer from RAG docs + search results + LLM."""
    llm = _get_llm()

    last_msg = state["messages"][-1]
    question = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    user_prompt = ACADEMIC_ANSWER_PROMPT.format(
        retrieved_context=_format_retrieved(state.get("retrieved_docs", [])),
        search_context=_format_search(state.get("search_results", [])),
        question=question,
    )

    fallback = get_fallback_llm(temperature=0.7)
    messages = [
        SystemMessage(content=ACADEMIC_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_answer",
        temperature=0.7,
    ) as span:
        response = invoke_with_fallback(
            llm, messages, fallback=fallback, span=span,
        )

    return {"messages": [AIMessage(content=response.content)]}
