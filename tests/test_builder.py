"""Unit tests for graph construction and compilation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.graph.builder import build_graph, get_compiled_graph


class TestBuildGraph:

    def test_returns_state_graph(self):
        from langgraph.graph import StateGraph
        graph = build_graph()
        assert isinstance(graph, StateGraph)

    def test_graph_has_all_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "supervisor",
            "academic_router",
            "rag_retrieve",
            "web_search",
            "generate_answer",
            "evaluate_hallucination",
            "search_policy",
            "generate_plan",
            "emotional_response",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_graph_compiles_without_error(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_get_compiled_graph_returns_compiled(self):
        compiled = get_compiled_graph()
        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "stream")
