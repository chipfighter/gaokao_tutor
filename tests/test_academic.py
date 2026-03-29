"""Unit tests for SubGraph A — Academic Tutor nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    _format_retrieved,
    _format_search,
    academic_router,
    generate_answer,
    rag_retrieve,
    rewrite_query,
    web_search,
)
from src.graph.state import CONTEXT_CLEAR


class TestAcademicRouterRetry:
    """academic_router clears context on retry path."""

    async def test_returns_empty_on_first_run(self):
        """First run (retry_count=0): no context clearing."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 0,
        }
        result = await academic_router(state)
        assert "context" not in result

    async def test_clears_context_on_retry(self):
        """On retry (retry_count > 0): returns CONTEXT_CLEAR to reset context."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 1,
            "context": [{"type": "rag", "content": "stale"}],
        }
        result = await academic_router(state)
        assert result["context"] is CONTEXT_CLEAR

    async def test_clears_context_on_second_retry(self):
        """retry_count=2 also triggers context clearing."""
        state = {
            "messages": [HumanMessage(content="test")],
            "retry_count": 2,
        }
        result = await academic_router(state)
        assert result["context"] is CONTEXT_CLEAR


class TestRewriteQuery:
    """rewrite_query node rewrites the user's question on retry."""

    @patch("src.graph.academic.get_node_llm")
    async def test_produces_rewritten_query(self, mock_get_llm):
        """Should call LLM and store result in rewritten_query."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="改进后的问题：判别式的具体用法"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "hallucination_reason": "答案未基于上下文",
            "retry_count": 1,
        }
        result = await rewrite_query(state)

        assert "rewritten_query" in result
        assert len(result["rewritten_query"]) > 0
        mock_get_llm.assert_called_once_with("supervisor")

    @patch("src.graph.academic.get_node_llm")
    async def test_uses_hallucination_reason_in_prompt(self, mock_get_llm):
        """The LLM prompt should include the hallucination reason."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="rewritten"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="original question")],
            "hallucination_reason": "fabricated formula",
            "retry_count": 1,
        }
        await rewrite_query(state)

        call_args = mock_llm.ainvoke.call_args[0][0]
        prompt_text = " ".join(m.content for m in call_args)
        assert "fabricated formula" in prompt_text
        assert "original question" in prompt_text

    @patch("src.graph.academic.get_node_llm")
    async def test_falls_back_to_original_on_failure(self, mock_get_llm):
        """On LLM failure, rewritten_query should be the original question."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="original question")],
            "hallucination_reason": "bad",
            "retry_count": 1,
        }
        result = await rewrite_query(state)

        assert result["rewritten_query"] == "original question"


class TestRagRetrieveWithRewrittenQuery:
    """rag_retrieve uses rewritten_query when available."""

    @patch("src.graph.academic.retrieve")
    async def test_uses_rewritten_query(self, mock_retrieve):
        """When rewritten_query is set, use it instead of keypoints."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="original")],
            "keypoints": ["original"],
            "subject": "math",
            "rewritten_query": "improved question about discriminant",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(
            query="improved question about discriminant", subject="math",
        )

    @patch("src.graph.academic.retrieve")
    async def test_ignores_empty_rewritten_query(self, mock_retrieve):
        """When rewritten_query is empty, fall back to keypoints."""
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="原始问题")],
            "keypoints": ["判别式"],
            "subject": "math",
            "rewritten_query": "",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="判别式", subject="math")


class TestWebSearchWithRewrittenQuery:
    """web_search uses rewritten_query when available."""

    @patch("src.graph.academic.web_search_fn")
    async def test_uses_rewritten_query(self, mock_search):
        """When rewritten_query is set, use it for web search."""
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="original")],
            "rewritten_query": "improved search query",
        }
        await web_search(state)

        mock_search.assert_called_once_with("improved search query")

    @patch("src.graph.academic.web_search_fn")
    async def test_ignores_empty_rewritten_query(self, mock_search):
        """When rewritten_query is empty, fall back to last human message."""
        mock_search.return_value = []

        state = {
            "messages": [HumanMessage(content="the real question")],
            "rewritten_query": "",
        }
        await web_search(state)

        mock_search.assert_called_once_with("the real question")


class TestRagRetrieve:

    @patch("src.graph.academic.retrieve")
    async def test_uses_keypoints_as_query(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": [{"content": "test", "source": "f.pdf", "score": 0.9}]}

        state = {
            "messages": [HumanMessage(content="什么是判别式")],
            "keypoints": ["二次函数", "判别式"],
            "subject": "math",
        }
        result = await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="二次函数 判别式", subject="math")
        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "rag"

    @patch("src.graph.academic.retrieve")
    async def test_falls_back_to_message_when_no_keypoints(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="告诉我关于椭圆的知识")],
            "keypoints": [],
            "subject": "math",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="告诉我关于椭圆的知识", subject="math")

    @patch("src.graph.academic.retrieve")
    async def test_subject_other_passes_none(self, mock_retrieve):
        mock_retrieve.return_value = {"docs": []}

        state = {
            "messages": [HumanMessage(content="test")],
            "keypoints": ["test"],
            "subject": "other",
        }
        await rag_retrieve(state)

        mock_retrieve.assert_called_once_with(query="test", subject=None)


class TestWebSearch:

    @patch("src.graph.academic.web_search_fn")
    async def test_returns_context_results(self, mock_search):
        mock_search.return_value = [{"content": "result", "title": "t", "url": "u"}]

        state = {"messages": [HumanMessage(content="量子力学")]}
        result = await web_search(state)

        assert len(result["context"]) == 1
        assert result["context"][0]["type"] == "web"
        mock_search.assert_called_once_with("量子力学")

    @patch("src.graph.academic.web_search_fn", side_effect=Exception("network error"))
    async def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = await web_search(state)

        assert result["context"] == []


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

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_generates_ai_message(self, mock_get_llm, mock_get_fallback, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("判别式 Δ=b²-4ac 的作用是..."))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "context": [{"type": "rag", "content": "Δ=b²-4ac", "source": "test.pdf", "score": 0.9}],
        }
        result = await generate_answer(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "判别式" in result["messages"][0].content

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    async def test_handles_empty_context(self, mock_get_llm, mock_get_fallback, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("I can help with that."))
        mock_get_llm.return_value = mock_llm
        mock_get_fallback.return_value = MagicMock()

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }
        result = await generate_answer(state)

        assert len(result["messages"]) == 1
