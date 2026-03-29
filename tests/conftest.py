"""Shared pytest fixtures for Gaokao Tutor unit tests.

All unit tests mock external dependencies (LLM APIs, ChromaDB, web search)
so they run offline without API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def human_msg():
    """Factory fixture for creating HumanMessage objects."""
    def _make(content: str = "你好") -> HumanMessage:
        return HumanMessage(content=content)
    return _make


@pytest.fixture
def ai_msg():
    """Factory fixture for creating AIMessage objects."""
    def _make(content: str = "你好，同学！") -> AIMessage:
        return AIMessage(content=content)
    return _make


@pytest.fixture
def sample_state(human_msg):
    """Minimal TutorState dict for testing."""
    return {
        "messages": [human_msg("二次函数的判别式怎么用？")],
        "intent": "academic",
        "subject": "math",
        "keypoints": ["二次函数", "判别式"],
        "context": [],
        "plan": "",
        "retry_count": 0,
        "hallucination_detected": False,
        "rewritten_query": "",
        "hallucination_reason": "",
    }


@pytest.fixture
def mock_llm_response():
    """Factory that creates a mock LLM response with given content."""
    def _make(content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        return resp
    return _make


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents for RAG tests."""
    return [
        {"content": "判别式 Δ=b²-4ac 用于判断二次方程根的情况。", "source": "math_2024.pdf", "score": 0.85, "metadata": {"subject": "math"}},
        {"content": "当 Δ>0 时有两个不等实根。", "source": "math_2024.pdf", "score": 0.72, "metadata": {"subject": "math"}},
    ]


@pytest.fixture
def sample_search_results():
    """Sample web search results."""
    return [
        {"content": "2026年高考时间为6月7日-8日。", "title": "高考时间", "url": "https://example.com/1"},
        {"content": "新高考改革3+1+2模式。", "title": "高考改革", "url": "https://example.com/2"},
    ]


def _reset_trace_provider():
    """Force-reset the global TracerProvider so tests can set their own."""
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None


@pytest.fixture
def in_memory_exporter():
    """Provide an InMemorySpanExporter for capturing spans in tests."""
    _reset_trace_provider()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()
    _reset_trace_provider()
