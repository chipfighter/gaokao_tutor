"""Unit tests for SubGraph A — Academic Tutor nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    _format_retrieved,
    _format_search,
    generate_answer,
    rag_retrieve,
    should_web_search,
    web_search,
)


class TestRagRetrieve:

    @patch("src.graph.academic.retrieve")
    def test_uses_keypoints_as_query(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": [{"content": "test", "source": "f.pdf", "score": 0.9}]}

        state = {
            "messages": [HumanMessage(content="什么是判别式")],
            "keypoints": ["二次函数", "判别式"],
            "subject": "math",
        }
        result = rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="二次函数 判别式", subject="math")
        assert len(result["retrieved_docs"]) == 1

    @patch("src.graph.academic.retrieve")
    def test_falls_back_to_message_when_no_keypoints(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="告诉我关于椭圆的知识")],
            "keypoints": [],
            "subject": "math",
        }
        rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="告诉我关于椭圆的知识", subject="math")

    @patch("src.graph.academic.retrieve")
    def test_subject_other_passes_none(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "other",
        }
        rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="test", subject=None)


class TestShouldWebSearch:

    def test_triggers_search_when_no_docs(self):
        state = {"retrieved_docs": []}
        assert should_web_search(state) == "web_search"

    def test_triggers_search_when_low_score(self):
        state = {"retrieved_docs": [{"score": 0.1}]}
        assert should_web_search(state) == "web_search"

    def test_skips_search_when_hit(self):
        state = {"retrieved_docs": [{"score": 0.85}]}
        assert should_web_search(state) == "generate_answer"

    def test_boundary_score_at_threshold(self):
        from src.rag.retriever import RELEVANCE_THRESHOLD
        state = {"retrieved_docs": [{"score": RELEVANCE_THRESHOLD}]}
        assert should_web_search(state) == "generate_answer"

    def test_boundary_score_below_threshold(self):
        from src.rag.retriever import RELEVANCE_THRESHOLD
        state = {"retrieved_docs": [{"score": RELEVANCE_THRESHOLD - 0.01}]}
        assert should_web_search(state) == "web_search"


class TestWebSearch:

    @patch("src.graph.academic.web_search_fn")
    def test_returns_search_results(self, mock_search):
        mock_search.return_value = [{"content": "result", "title": "t", "url": "u"}]

        state = {"messages": [HumanMessage(content="量子力学")]}
        result = web_search(state)

        assert len(result["search_results"]) == 1
        mock_search.assert_called_once_with("量子力学")

    @patch("src.graph.academic.web_search_fn", side_effect=Exception("network error"))
    def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = web_search(state)

        assert result["search_results"] == []


class TestFormatHelpers:

    def test_format_retrieved_empty(self):
        assert _format_retrieved([]) == "无相关参考资料。"

    def test_format_retrieved_with_docs(self, sample_retrieved_docs):
        output = _format_retrieved(sample_retrieved_docs)
        assert "[1]" in output
        assert "[2]" in output
        assert "math_2024.pdf" in output

    def test_format_search_empty(self):
        assert _format_search([]) == "无网络搜索结果。"

    def test_format_search_with_results(self, sample_search_results):
        output = _format_search(sample_search_results)
        assert "[1]" in output
        assert "高考时间" in output


class TestGenerateAnswer:

    @patch("src.graph.academic._get_llm")
    def test_generates_ai_message(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("判别式 Δ=b²-4ac 的作用是...")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "retrieved_docs": [{"content": "Δ=b²-4ac", "source": "test.pdf", "score": 0.9}],
            "search_results": [],
        }
        result = generate_answer(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "判别式" in result["messages"][0].content

    @patch("src.graph.academic._get_llm")
    def test_handles_empty_context(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("I can help with that.")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="test")],
            "retrieved_docs": [],
            "search_results": [],
        }
        result = generate_answer(state)

        assert len(result["messages"]) == 1
