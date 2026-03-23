"""SubGraph A — Academic Tutor: parallel retrieval (fan-out/fan-in),
answer generation, and hallucination evaluation with retry loop.

Keypoint extraction is handled by the supervisor node (merged for latency),
so this subgraph starts at the academic_router which fans out to both
rag_retrieve and web_search in parallel.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_setting, load_prompt
from src.graph.llm import get_fallback_llm, get_node_llm, invoke_with_fallback
from src.graph.state import TutorState
from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn
from src.tracing import traced_llm_call, traced_node, traced_retrieval, traced_search

logger = logging.getLogger(__name__)

MAX_RETRIES = get_setting("academic.max_retries", 2)


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


def _last_human_query(state: TutorState) -> str:
    """Extract the last HumanMessage content (robust for retry loops)."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ── Node 0: academic router (fan-out trigger) ─────────────────────

@traced_node
def academic_router(state: TutorState) -> dict:
    """No-op router node that enables parallel fan-out to retrieval nodes."""
    return {}


# ── Node 1: RAG retrieval (parallel branch A) ─────────────────────

@traced_node
def rag_retrieve(state: TutorState) -> dict:
    """Query ChromaDB with keypoints extracted by the supervisor node."""
    keypoints = state.get("keypoints", [])
    subject = state.get("subject")
    query = " ".join(keypoints) if keypoints else _last_human_query(state)

    subj = subject if subject != "other" else None

    with traced_retrieval(query=query, subject=subj) as span:
        result = retrieve(query=query, subject=subj)
        span.set_attribute("rag.doc_count", len(result.get("docs", [])))
        span.set_attribute("rag.is_hit", result.get("is_hit", False))
        if result.get("docs"):
            span.set_attribute("rag.top_score", result["docs"][0].get("score", 0))

    docs = result["docs"]
    return {"context": [{"type": "rag", **doc} for doc in docs]}


# ── Node 2: web search (parallel branch B) ────────────────────────

_SEARCH_TIMEOUT = get_setting("academic.search_timeout", 15)


@traced_node
def web_search(state: TutorState) -> dict:
    """Fan-out web search — runs in parallel with rag_retrieve."""
    query = _last_human_query(state)

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

    return {"context": [{"type": "web", **r} for r in search_results]}


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
    """Synthesize final answer from merged context (RAG + web) via LLM."""
    llm = get_node_llm("academic")

    question = _last_human_query(state)

    # Split merged context by source type
    context = state.get("context", [])
    rag_docs = [c for c in context if c.get("type") == "rag"]
    web_results = [c for c in context if c.get("type") == "web"]

    temperature = get_setting("academic.temperature", 0.7)
    user_prompt = load_prompt("academic_answer").format(
        retrieved_context=_format_retrieved(rag_docs),
        search_context=_format_search(web_results),
        question=question,
    )

    fallback = get_fallback_llm(temperature=temperature)
    messages = [
        SystemMessage(content=load_prompt("academic_system")),
        HumanMessage(content=user_prompt),
    ]

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="generate_answer",
        temperature=temperature,
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
    eval_temp = get_setting("academic.hallucination_eval_temperature", 0.0)
    llm = get_node_llm("academic", temperature=eval_temp)
    structured_primary = llm.with_structured_output(HallucinationEvaluation)

    fallback_llm = get_fallback_llm(temperature=eval_temp)
    structured_fallback = fallback_llm.with_structured_output(HallucinationEvaluation)

    # Extract the generated answer (last message) and original question
    answer = state["messages"][-1].content
    question = _last_human_query(state)

    # Build context from all retrieval sources
    docs = state.get("context", [])
    context = "\n".join(d.get("content", "") for d in docs) if docs else ""

    eval_prompt = load_prompt("hallucination_eval").format(
        question=question, context=context, answer=answer,
    )

    retry_count = state.get("retry_count", 0)

    with traced_llm_call(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        node_name="evaluate_hallucination",
        temperature=eval_temp,
    ) as span:
        try:
            evaluation = invoke_with_fallback(
                structured_primary,
                [
                    SystemMessage(content=load_prompt("hallucination_system")),
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
    """Conditional edge: retry via academic_router or route to END.

    Allows up to MAX_RETRIES re-retrieval attempts when hallucination
    is detected. After exhausting retries, routes to END regardless.
    """
    if (
        state.get("hallucination_detected", False)
        and state.get("retry_count", 0) <= MAX_RETRIES
    ):
        return "retry"
    return "end"
