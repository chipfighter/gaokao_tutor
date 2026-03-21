"""OpenTelemetry tracing for Gaokao Tutor."""

from src.tracing.collector import get_tracer, setup_tracing, shutdown_tracing
from src.tracing.decorators import (
    traced_llm_call,
    traced_node,
    traced_retrieval,
    traced_search,
)

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "get_tracer",
    "traced_node",
    "traced_llm_call",
    "traced_retrieval",
    "traced_search",
]
