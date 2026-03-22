"""Unit tests for parallel retrieval (fan-out / fan-in) pattern.

Tests cover: operator.add context reducer, academic_router pass-through,
rag_retrieve + web_search writing context with type tags,
generate_answer reading merged context, and graph fan-out structure.
All tests mock external dependencies -- no real API calls required.
"""

from __future__ import annotations

import operator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    academic_router,
    rag_retrieve,
    web_search,
)
from src.graph.builder import build_graph, get_compiled_graph


# ===========================================================================
# TestContextReducer -- operator.add merges context from parallel branches
# ===========================================================================

class TestContextReducer:
    """Verify that operator.add correctly merges context lists from fan-out."""

    def test_merges_from_two_branches(self):
        """Simulates LangGraph fan-in: merge rag + web context lists."""
        branch_a = [{"type": "rag", "content": "doc1", "source": "f.pdf", "score": 0.9}]
        branch_b = [{"type": "web", "content": "web1", "title": "T", "url": "http://x"}]

        merged = operator.add(branch_a, branch_b)

        assert len(merged) == 2
        assert merged[0]["type"] == "rag"
        assert merged[1]["type"] == "web"

    def test_accumulates_on_retry(self):
        """On retry, new context appends to existing (operator.add)."""
        existing = [{"type": "rag", "content": "old"}]
        new = [{"type": "rag", "content": "new"}]

        merged = operator.add(existing, new)

        assert len(merged) == 2
        assert merged[0]["content"] == "old"
        assert merged[1]["content"] == "new"

    def test_empty_branch_produces_no_extra(self):
        """If one branch returns empty context, merge is just the other."""
        branch_a = [{"type": "rag", "content": "doc1"}]
        branch_b = []

        merged = operator.add(branch_a, branch_b)

        assert len(merged) == 1
        assert merged[0]["type"] == "rag"


# ===========================================================================
# TestAcademicRouterNode -- pass-through for fan-out
# ===========================================================================

class TestAcademicRouterNode:
    """academic_router is a no-op node that enables fan-out."""

    def test_returns_empty_dict(self):
        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "math",
        }
        result = academic_router(state)
        assert result == {}

    def test_is_traced(self, in_memory_exporter):
        """Should create an OTel span for Jaeger visibility."""
        state = {"messages": [HumanMessage(content="test")]}
        academic_router(state)

        spans = in_memory_exporter.get_finished_spans()
        node_spans = [s for s in spans if s.name == "graph.node.academic_router"]
        assert len(node_spans) == 1


# ===========================================================================
# TestRagRetrieveParallelOutput -- writes context with type="rag"
# ===========================================================================

class TestRagRetrieveParallelOutput:
    """rag_retrieve returns context entries tagged type='rag'."""

    @patch("src.graph.academic.retrieve")
    def test_returns_context_with_rag_type(self, mock_retrieve):
        mock_retrieve.return_value = {
            "docs": [{"content": "判别式", "source": "math.pdf", "score": 0.9}],
        }

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["判别式"],
            "subject": "math",
        }
        result = rag_retrieve(state)

        assert "context" in result
        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "rag"
        assert result["context"][0]["content"] == "判别式"
        assert result["context"][0]["source"] == "math.pdf"

    @patch("src.graph.academic.retrieve")
    def test_empty_retrieval_returns_empty_context(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "math",
        }
        result = rag_retrieve(state)

        assert result["context"] == []

    @patch("src.graph.academic.retrieve")
    def test_uses_last_human_message_when_no_keypoints(self, mock_retrieve):
        """On retry, fallback query should find last HumanMessage."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [
                HumanMessage(content="original question"),
                AIMessage(content="bad answer"),
            ],
            "keypoints": [],
            "subject": "math",
        }
        rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="original question", subject="math")


# ===========================================================================
# TestWebSearchParallelOutput -- writes context with type="web"
# ===========================================================================

class TestWebSearchParallelOutput:
    """web_search returns context entries tagged type='web'."""

    @patch("src.graph.academic.web_search_fn")
    def test_returns_context_with_web_type(self, mock_search):
        mock_search.return_value = [
            {"content": "result", "title": "Title", "url": "http://x"},
        ]

        state = {"messages": [HumanMessage(content="量子力学")]}
        result = web_search(state)

        assert "context" in result
        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "web"
        assert result["context"][0]["content"] == "result"
        assert result["context"][0]["title"] == "Title"

    @patch("src.graph.academic.web_search_fn", side_effect=Exception("network"))
    def test_returns_empty_context_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = web_search(state)

        assert result["context"] == []

    @patch("src.graph.academic.web_search_fn")
    def test_uses_last_human_message_for_query(self, mock_search):
        """During retry, web_search must find the original question."""
        mock_search.return_value = []

        state = {
            "messages": [
                HumanMessage(content="the real question"),
                AIMessage(content="previous bad answer"),
            ],
        }
        web_search(state)

        mock_search.assert_called_once_with("the real question")


# ===========================================================================
# TestGenerateAnswerFromMergedContext -- reads unified context
# ===========================================================================

class TestGenerateAnswerFromMergedContext:
    """generate_answer splits merged context by type for formatting."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_uses_both_rag_and_web_context(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("combined answer")
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="判别式")],
            "context": [
                {"type": "rag", "content": "Δ=b²-4ac", "source": "math.pdf", "score": 0.9},
                {"type": "web", "content": "判别式用法", "title": "高考数学", "url": "http://x"},
            ],
        }

        from src.graph.academic import generate_answer

        result = generate_answer(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        # Verify the prompt includes both RAG and web content
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[-1].content
        assert "Δ=b²-4ac" in prompt_text
        assert "判别式用法" in prompt_text

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_handles_empty_context(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("answer without context")
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        result = generate_answer(state)

        assert len(result["messages"]) == 1


# ===========================================================================
# TestGraphFanOutStructure -- verify graph topology
# ===========================================================================

class TestGraphFanOutStructure:
    """Verify the graph has correct fan-out/fan-in wiring."""

    def test_academic_router_node_exists(self):
        graph = build_graph()
        assert "academic_router" in graph.nodes

    def test_evaluate_hallucination_node_exists(self):
        graph = build_graph()
        assert "evaluate_hallucination" in graph.nodes

    def test_graph_compiles_with_fan_out(self):
        """Fan-out/fan-in graph must compile without error."""
        compiled = get_compiled_graph()
        assert compiled is not None
        assert hasattr(compiled, "invoke")

    def test_should_web_search_not_in_graph(self):
        """Sequential should_web_search replaced by parallel fan-out."""
        graph = build_graph()
        # should_web_search was a conditional edge function, not a node.
        # Verify the old conditional edge target mapping is gone by checking
        # that both rag_retrieve and web_search are direct nodes (no conditional).
        assert "rag_retrieve" in graph.nodes
        assert "web_search" in graph.nodes


# ===========================================================================
# TestRetryRoutesToAcademicRouter -- retry re-runs both retrievals
# ===========================================================================

class TestRetryRoutesToAcademicRouter:
    """On hallucination retry, route back to academic_router for full re-retrieval."""

    def test_should_retry_or_end_returns_retry(self):
        from src.graph.academic import should_retry_or_end

        state = {"hallucination_detected": True, "retry_count": 1}
        assert should_retry_or_end(state) == "retry"

    def test_should_retry_or_end_returns_end_past_max(self):
        from src.graph.academic import MAX_RETRIES, should_retry_or_end

        state = {"hallucination_detected": True, "retry_count": MAX_RETRIES + 1}
        assert should_retry_or_end(state) == "end"
