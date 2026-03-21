"""Unit tests for SSE node lifecycle events in generate_sse.

Tests cover: node start/end events, token streaming coexistence,
sub-chain filtering, internal node filtering, and event ordering.
All tests mock astream_events — no real graph execution required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: reusable async iterator from a list
# ---------------------------------------------------------------------------

class AsyncIteratorMock:
    """Create an async iterator from a list of items."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# Helpers: build mock events matching astream_events v2 format
# ---------------------------------------------------------------------------

def _node_start(node_name: str) -> dict:
    """Build an on_chain_start event for a graph node."""
    return {
        "event": "on_chain_start",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"input": {}},
    }


def _node_end(node_name: str) -> dict:
    """Build an on_chain_end event for a graph node."""
    return {
        "event": "on_chain_end",
        "name": node_name,
        "metadata": {"langgraph_node": node_name},
        "data": {"output": {}},
    }


def _sub_chain_start(chain_name: str, parent_node: str) -> dict:
    """Build an on_chain_start for an internal sub-chain (not a graph node)."""
    return {
        "event": "on_chain_start",
        "name": chain_name,
        "metadata": {"langgraph_node": parent_node},
        "data": {"input": {}},
    }


def _token_event(node_name: str, content: str) -> dict:
    """Build an on_chat_model_stream event with a token chunk."""
    chunk = SimpleNamespace(content=content)
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"chunk": chunk},
    }


# ---------------------------------------------------------------------------
# TestSSENodeLifecycle
# ---------------------------------------------------------------------------

class TestSSENodeLifecycle:
    """Tests that generate_sse emits node lifecycle events."""

    @pytest.mark.anyio
    async def test_yields_node_start_event(self):
        """on_chain_start for a graph node → {"type": "node_event", "status": "start"}."""
        from app import generate_sse

        events = [_node_start("supervisor")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hello"):
                collected.append(sse)

        assert len(collected) == 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data == {"type": "node_event", "status": "start", "node": "supervisor"}

    @pytest.mark.anyio
    async def test_yields_node_end_event(self):
        """on_chain_end for a graph node → {"type": "node_event", "status": "end"}."""
        from app import generate_sse

        events = [_node_end("rag_retrieve")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hello"):
                collected.append(sse)

        assert len(collected) == 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data == {"type": "node_event", "status": "end", "node": "rag_retrieve"}

    @pytest.mark.anyio
    async def test_ignores_sub_chain_events(self):
        """Sub-chain events (name != metadata.langgraph_node) must be dropped."""
        from app import generate_sse

        events = [
            _sub_chain_start("RunnableSequence", "supervisor"),
            _sub_chain_start("ChatPromptTemplate", "generate_answer"),
        ]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hello"):
                collected.append(sse)

        assert collected == []

    @pytest.mark.anyio
    async def test_ignores_langgraph_internal_nodes(self):
        """LangGraph internal nodes like __start__ must be filtered out."""
        from app import generate_sse

        events = [
            {
                "event": "on_chain_start",
                "name": "__start__",
                "metadata": {"langgraph_node": "__start__"},
                "data": {},
            },
        ]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hello"):
                collected.append(sse)

        assert collected == []


# ---------------------------------------------------------------------------
# TestSSETokenStreamingPreserved
# ---------------------------------------------------------------------------

class TestSSETokenStreamingPreserved:
    """Ensure the original token streaming logic is not broken."""

    @pytest.mark.anyio
    async def test_token_from_allowed_node(self):
        """Tokens from ALLOWED_NODES should still be emitted."""
        from app import generate_sse

        events = [_token_event("generate_answer", "Hello")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hi"):
                collected.append(sse)

        assert len(collected) == 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data == {"type": "token", "content": "Hello"}

    @pytest.mark.anyio
    async def test_token_from_disallowed_node_dropped(self):
        """Tokens from nodes NOT in ALLOWED_NODES should be dropped."""
        from app import generate_sse

        events = [_token_event("supervisor", "thinking...")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hi"):
                collected.append(sse)

        assert collected == []

    @pytest.mark.anyio
    async def test_empty_token_dropped(self):
        """Empty token content should not produce an SSE payload."""
        from app import generate_sse

        events = [_token_event("generate_answer", "")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("hi"):
                collected.append(sse)

        assert collected == []


# ---------------------------------------------------------------------------
# TestSSEMixedEventOrdering
# ---------------------------------------------------------------------------

class TestSSEMixedEventOrdering:
    """Tests correct ordering when lifecycle and token events are interleaved."""

    @pytest.mark.anyio
    async def test_full_academic_flow(self):
        """Simulate supervisor → rag_retrieve → generate_answer with tokens."""
        from app import generate_sse

        events = [
            _node_start("supervisor"),
            _sub_chain_start("RunnableSequence", "supervisor"),
            _node_end("supervisor"),
            _node_start("rag_retrieve"),
            _node_end("rag_retrieve"),
            _node_start("generate_answer"),
            _token_event("generate_answer", "The"),
            _token_event("generate_answer", " answer"),
            _node_end("generate_answer"),
        ]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("What is calculus?"):
                collected.append(sse)

        # Sub-chain event should be dropped → 8 events minus 1 = 7
        # node_start(supervisor), node_end(supervisor),
        # node_start(rag), node_end(rag),
        # node_start(gen), token, token, node_end(gen) = 8
        assert len(collected) == 8

        payloads = [
            json.loads(s.removeprefix("data: ").strip()) for s in collected
        ]

        assert payloads[0] == {"type": "node_event", "status": "start", "node": "supervisor"}
        assert payloads[1] == {"type": "node_event", "status": "end", "node": "supervisor"}
        assert payloads[2] == {"type": "node_event", "status": "start", "node": "rag_retrieve"}
        assert payloads[3] == {"type": "node_event", "status": "end", "node": "rag_retrieve"}
        assert payloads[4] == {"type": "node_event", "status": "start", "node": "generate_answer"}
        assert payloads[5] == {"type": "token", "content": "The"}
        assert payloads[6] == {"type": "token", "content": " answer"}
        assert payloads[7] == {"type": "node_event", "status": "end", "node": "generate_answer"}

    @pytest.mark.anyio
    async def test_emotional_flow(self):
        """Simulate supervisor → emotional_response with tokens."""
        from app import generate_sse

        events = [
            _node_start("supervisor"),
            _node_end("supervisor"),
            _node_start("emotional_response"),
            _token_event("emotional_response", "I understand"),
            _node_end("emotional_response"),
        ]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("I'm stressed"):
                collected.append(sse)

        assert len(collected) == 5
        payloads = [
            json.loads(s.removeprefix("data: ").strip()) for s in collected
        ]
        assert payloads[0]["type"] == "node_event"
        assert payloads[3]["type"] == "token"
        assert payloads[3]["content"] == "I understand"


# ---------------------------------------------------------------------------
# TestSSEAllGraphNodes
# ---------------------------------------------------------------------------

class TestSSEAllGraphNodes:
    """Ensure every known graph node can produce lifecycle events."""

    ALL_NODES = [
        "supervisor",
        "rag_retrieve",
        "web_search",
        "generate_answer",
        "search_policy",
        "generate_plan",
        "emotional_response",
    ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_start(self, node_name):
        """Every graph node should produce a start event."""
        from app import generate_sse

        events = [_node_start(node_name)]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        assert len(collected) == 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data["node"] == node_name
        assert data["status"] == "start"

    @pytest.mark.anyio
    @pytest.mark.parametrize("node_name", ALL_NODES)
    async def test_each_node_emits_end(self, node_name):
        """Every graph node should produce an end event."""
        from app import generate_sse

        events = [_node_end(node_name)]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        assert len(collected) == 1
        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data["node"] == node_name
        assert data["status"] == "end"
