"""Tracing decorators -- non-intrusive instrumentation for graph nodes and operations."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.tracing.collector import get_tracer


# ---------------------------------------------------------------------------
# @traced_node — wraps LangGraph node functions
# ---------------------------------------------------------------------------

def traced_node(func: Callable) -> Callable:
    """Decorator that wraps a LangGraph node function with an OpenTelemetry span.

    Records: node name, execution duration, input/output state keys,
    and any exceptions as span events.
    """

    @functools.wraps(func)
    def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            f"graph.node.{func.__name__}",
            attributes={
                "graph.node.name": func.__name__,
                "graph.node.input_keys": str(list(state.keys())),
            },
        ) as span:
            try:
                result = func(state, *args, **kwargs)

                if isinstance(result, dict):
                    span.set_attribute("graph.node.output_keys", str(list(result.keys())))

                    if "intent" in result:
                        span.set_attribute("graph.node.intent", result["intent"])
                    if "subject" in result:
                        span.set_attribute("graph.node.subject", result["subject"])
                    if "keypoints" in result:
                        span.set_attribute("graph.node.keypoint_count", len(result["keypoints"]))
                    if "context" in result:
                        span.set_attribute("graph.node.context_count", len(result["context"]))
                    if "search_results" in result:
                        span.set_attribute("graph.node.search_result_count", len(result["search_results"]))
                    if "messages" in result:
                        span.set_attribute("graph.node.message_count", len(result["messages"]))
                    if "retry_count" in result:
                        span.set_attribute("graph.node.retry_count", result["retry_count"])
                    if "hallucination_detected" in result:
                        span.set_attribute("graph.node.hallucination_detected", result["hallucination_detected"])

                span.set_status(StatusCode.OK)
                return result

            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# traced_llm_call — context manager for LLM invocations
# ---------------------------------------------------------------------------

@contextmanager
def traced_llm_call(
    model_name: str = "unknown",
    node_name: str = "unknown",
    temperature: float | None = None,
) -> Generator[trace.Span, None, None]:
    """Context manager that creates a child span for an LLM invocation.

    Yields the span so callers can set additional attributes (e.g. token counts).
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"llm.invoke.{node_name}",
        attributes={
            "llm.model": model_name,
            "llm.node": node_name,
        },
    ) as span:
        if temperature is not None:
            span.set_attribute("llm.temperature", temperature)

        start = time.monotonic()
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            span.set_attribute("llm.latency_ms", round(latency_ms, 2))


# ---------------------------------------------------------------------------
# traced_retrieval — context manager for RAG retrieval
# ---------------------------------------------------------------------------

@contextmanager
def traced_retrieval(
    query: str,
    subject: str | None = None,
    top_k: int = 5,
) -> Generator[trace.Span, None, None]:
    """Context manager for RAG retrieval spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "rag.retrieve",
        attributes={
            "rag.query": query[:200],
            "rag.subject": subject or "all",
            "rag.top_k": top_k,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


# ---------------------------------------------------------------------------
# traced_search — context manager for web search
# ---------------------------------------------------------------------------

@contextmanager
def traced_search(
    query: str,
    timeout: int = 15,
) -> Generator[trace.Span, None, None]:
    """Context manager for web search spans."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "web.search",
        attributes={
            "search.query": query[:200],
            "search.timeout_sec": timeout,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
