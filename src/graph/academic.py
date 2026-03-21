"""SubGraph A — Academic Tutor: RAG retrieval, web search fallback, answer generation,
and hallucination evaluation with retry loop.

Keypoint extraction is handled by the supervisor node (merged for latency),
so this subgraph starts directly at RAG retrieval.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.graph.llm import get_fallback_llm, invoke_with_fallback
from src.graph.state import TutorState
from src.prompts.academic import (
    ACADEMIC_ANSWER_PROMPT,
    ACADEMIC_SYSTEM_PROMPT,
    HALLUCINATION_EVAL_PROMPT,
    HALLUCINATION_SYSTEM_PROMPT,
)
from src.rag.retriever import RELEVANCE_THRESHOLD, retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── Structured output schema for hallucination evaluation ─────────
class HallucinationEvaluation(BaseModel):
    """LLM-evaluated faithfulness judgment."""

    is_faithful: bool = Field(
        description="True if the answer is grounded in the retrieved context "
        "and addresses the student's question without fabrication",
    )
    reason: str = Field(
        description="Brief explanation of the evaluation judgment",
    )


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

    # Find the original question (last HumanMessage, robust for retry loops)
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

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


# ── Node 4: hallucination evaluation (reflection loop) ─────────

@traced_node
def evaluate_hallucination(state: TutorState) -> dict:
    """Evaluate whether the generated answer hallucinates beyond retrieved context.

    Uses structured LLM output to judge faithfulness. On detection,
    increments retry_count to signal the conditional edge for re-retrieval.
    Defaults to valid on any parsing/model failure (safe fallback).
    """
    llm = _get_llm(temperature=0.0)
    structured_primary = llm.with_structured_output(HallucinationEvaluation)

    fallback_llm = get_fallback_llm(temperature=0.0)
    structured_fallback = fallback_llm.with_structured_output(HallucinationEvaluation)

    # Extract the generated answer (last message) and original question
    answer = state["messages"][-1].content
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # Build context from retrieved docs
    docs = state.get("retrieved_docs", [])
    context = "\n".join(d.get("content", "") for d in docs) if docs else ""

    eval_prompt = HALLUCINATION_EVAL_PROMPT.format(
        question=question, context=context, answer=answer,
    )

    retry_count = state.get("retry_count", 0)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=0.0,
    ) as span:
        try:
            evaluation = invoke_with_fallback(
                structured_primary,
                [
                    SystemMessage(content=HALLUCINATION_SYSTEM_PROMPT),
                    HumanMessage(content=eval_prompt),
                ],
                fallback=structured_fallback,
                span=span,
            )
            is_faithful = evaluation.is_faithful
        except Exception:
            logger.warning("Hallucination evaluation failed, defaulting to valid")
            is_faithful = True

    hallucination_detected = not is_faithful

    result: dict = {"hallucination_detected": hallucination_detected}
    if hallucination_detected:
        result["retry_count"] = retry_count + 1

    return result


def should_retry_or_end(state: TutorState) -> str:
    """Conditional edge: retry via rag_retrieve or route to END.

    Allows up to MAX_RETRIES re-retrieval attempts when hallucination
    is detected. After exhausting retries, routes to END regardless.
    """
    if (
        state.get("hallucination_detected", False)
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "retry"
    return "end"
