"""Unit tests for tool wrappers (search_tool, rag_tool)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.search_tool import get_search_tool, search


class TestSearchFunction:

    @patch("src.tools.search_tool.get_search_tool")
    def test_returns_normalized_results(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = [
            {"snippet": "content1", "title": "title1", "link": "url1"},
            {"snippet": "content2", "title": "title2", "link": "url2"},
        ]
        mock_get_tool.return_value = mock_tool

        results = search("test query")

        assert len(results) == 2
        assert results[0]["content"] == "content1"
        assert results[0]["title"] == "title1"
        assert results[0]["url"] == "url1"

    @patch("src.tools.search_tool.get_search_tool")
    def test_handles_string_response(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "plain text result"
        mock_get_tool.return_value = mock_tool

        results = search("test query")

        assert len(results) == 1
        assert results[0]["content"] == "plain text result"

    @patch("src.tools.search_tool.get_search_tool")
    def test_returns_empty_on_exception(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = Exception("API error")
        mock_get_tool.return_value = mock_tool

        results = search("test query")

        assert results == []

    @patch("src.tools.search_tool.get_search_tool")
    def test_handles_alternate_key_names(self, mock_get_tool):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = [
            {"content": "c1", "title": "t1", "url": "u1"},
        ]
        mock_get_tool.return_value = mock_tool

        results = search("test")

        assert results[0]["content"] == "c1"
        assert results[0]["url"] == "u1"


class TestGetSearchTool:

    def test_creates_singleton(self):
        try:
            import ddgs  # noqa: F401
        except ImportError:
            pytest.skip("ddgs package not installed")

        import src.tools.search_tool as mod
        mod._search_tool = None

        tool = get_search_tool()
        assert tool is not None

        tool2 = get_search_tool()
        assert tool is tool2


class TestPrompts:
    """Verify prompt templates are well-formed and contain expected placeholders."""

    def test_supervisor_prompt_not_empty(self):
        from src.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
        assert len(SUPERVISOR_SYSTEM_PROMPT) > 100
        assert "academic" in SUPERVISOR_SYSTEM_PROMPT
        assert "planning" in SUPERVISOR_SYSTEM_PROMPT
        assert "emotional" in SUPERVISOR_SYSTEM_PROMPT

    def test_academic_prompts_have_placeholders(self):
        from src.prompts.academic import ACADEMIC_ANSWER_PROMPT, ACADEMIC_SYSTEM_PROMPT
        assert "{retrieved_context}" in ACADEMIC_ANSWER_PROMPT
        assert "{search_context}" in ACADEMIC_ANSWER_PROMPT
        assert "{question}" in ACADEMIC_ANSWER_PROMPT
        assert len(ACADEMIC_SYSTEM_PROMPT) > 50

    def test_planner_prompts_have_placeholders(self):
        from src.prompts.planner import PLANNER_GENERATE_PROMPT, PLANNER_SYSTEM_PROMPT
        assert "{user_request}" in PLANNER_GENERATE_PROMPT
        assert "{policy_info}" in PLANNER_GENERATE_PROMPT
        assert len(PLANNER_SYSTEM_PROMPT) > 50

    def test_emotional_prompt_not_empty(self):
        from src.prompts.emotional import EMOTIONAL_SYSTEM_PROMPT
        assert len(EMOTIONAL_SYSTEM_PROMPT) > 50
        assert "班主任" in EMOTIONAL_SYSTEM_PROMPT
