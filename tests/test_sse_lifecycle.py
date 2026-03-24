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
        assert data["type"] == "node_event"
        assert data["status"] == "end"
        assert data["node"] == "rag_retrieve"
        assert "duration_ms" in data
        assert data["error"] is None

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
        assert payloads[1]["type"] == "node_event"
        assert payloads[1]["status"] == "end"
        assert payloads[1]["node"] == "supervisor"
        assert payloads[2] == {"type": "node_event", "status": "start", "node": "rag_retrieve"}
        assert payloads[3]["type"] == "node_event"
        assert payloads[3]["status"] == "end"
        assert payloads[3]["node"] == "rag_retrieve"
        assert payloads[4] == {"type": "node_event", "status": "start", "node": "generate_answer"}
        assert payloads[5] == {"type": "token", "content": "The"}
        assert payloads[6] == {"type": "token", "content": " answer"}
        assert payloads[7]["type"] == "node_event"
        assert payloads[7]["status"] == "end"
        assert payloads[7]["node"] == "generate_answer"

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
        """Every graph node should produce an end event with duration_ms and error."""
        from app import generate_sse

        events = [_node_start(node_name), _node_end(node_name)]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        assert len(collected) == 2
        data = json.loads(collected[1].removeprefix("data: ").strip())
        assert data["node"] == node_name
        assert data["status"] == "end"
        assert isinstance(data["duration_ms"], int)
        assert data["error"] is None


# ---------------------------------------------------------------------------
# TestSSENodeTiming
# ---------------------------------------------------------------------------

class TestSSENodeTiming:
    """Tests that node end events include duration_ms."""

    @pytest.mark.anyio
    async def test_end_has_duration_ms(self):
        """A start+end pair should produce a non-negative duration_ms."""
        from app import generate_sse

        events = [_node_start("supervisor"), _node_end("supervisor")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        end_data = json.loads(collected[1].removeprefix("data: ").strip())
        assert end_data["duration_ms"] is not None
        assert end_data["duration_ms"] >= 0

    @pytest.mark.anyio
    async def test_end_without_start_has_null_duration(self):
        """An end event without a preceding start should have duration_ms=None."""
        from app import generate_sse

        events = [_node_end("supervisor")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        data = json.loads(collected[0].removeprefix("data: ").strip())
        assert data["duration_ms"] is None


# ---------------------------------------------------------------------------
# TestSSEErrorCapture
# ---------------------------------------------------------------------------

class TestSSEErrorCapture:
    """Tests that node end events capture errors."""

    @pytest.mark.anyio
    async def test_error_null_on_success(self):
        """Normal end → error is null."""
        from app import generate_sse

        events = [_node_start("supervisor"), _node_end("supervisor")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        data = json.loads(collected[1].removeprefix("data: ").strip())
        assert data["error"] is None

    @pytest.mark.anyio
    async def test_error_captured_from_output(self):
        """End event with error in output → error field populated."""
        from app import generate_sse

        end_event = {
            "event": "on_chain_end",
            "name": "web_search",
            "metadata": {"langgraph_node": "web_search"},
            "data": {"output": {"error": "TimeoutError: request timed out"}},
        }
        events = [_node_start("web_search"), end_event]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        data = json.loads(collected[1].removeprefix("data: ").strip())
        assert data["error"] == "TimeoutError: request timed out"


# ---------------------------------------------------------------------------
# TestSSEUsageEvents
# ---------------------------------------------------------------------------

def _chat_model_end(node_name: str, input_tokens: int, output_tokens: int, total_tokens: int) -> dict:
    """Build an on_chat_model_end event with usage_metadata."""
    output = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    )
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }


def _chat_model_end_no_usage(node_name: str) -> dict:
    """Build an on_chat_model_end event without usage_metadata."""
    output = SimpleNamespace(usage_metadata=None)
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node_name},
        "data": {"output": output},
    }


class TestSSEUsageEvents:
    """Tests for token usage SSE events."""

    @pytest.mark.anyio
    async def test_emits_usage_event(self):
        """on_chat_model_end with usage_metadata → usage SSE event."""
        from app import generate_sse

        events = [_chat_model_end("generate_answer", 100, 50, 150)]
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
        assert data == {
            "type": "usage",
            "node": "generate_answer",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    @pytest.mark.anyio
    async def test_no_usage_event_when_no_metadata(self):
        """on_chat_model_end without usage_metadata → no event emitted."""
        from app import generate_sse

        events = [_chat_model_end_no_usage("generate_answer")]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        assert collected == []

    @pytest.mark.anyio
    async def test_usage_interleaved_with_node_events(self):
        """Usage events appear alongside node lifecycle events."""
        from app import generate_sse

        events = [
            _node_start("generate_answer"),
            _token_event("generate_answer", "Hi"),
            _chat_model_end("generate_answer", 200, 100, 300),
            _node_end("generate_answer"),
        ]
        mock_graph = MagicMock()
        mock_graph.astream_events = MagicMock(
            return_value=AsyncIteratorMock(events),
        )

        collected = []
        with patch("app.graph", mock_graph):
            async for sse in generate_sse("q"):
                collected.append(sse)

        payloads = [json.loads(s.removeprefix("data: ").strip()) for s in collected]
        types = [p["type"] for p in payloads]
        assert types == ["node_event", "token", "usage", "node_event"]
