"""Unit tests for SubGraph B — Study Planner nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.planner import generate_plan, search_policy


class TestSearchPolicy:

    @patch("src.graph.planner.web_search_fn")
    def test_returns_search_results(self, mock_search):
        mock_search.return_value = [
            {"content": "2026年高考6月7日", "title": "高考时间", "url": "https://example.com"},
        ]

        state = {"messages": [HumanMessage(content="帮我做复习计划")]}
        result = search_policy(state)

        assert "search_results" in result
        assert len(result["search_results"]) == 1
        mock_search.assert_called_once()

    @patch("src.graph.planner.web_search_fn", side_effect=Exception("timeout"))
    def test_returns_empty_on_exception(self, mock_search):
        state = {"messages": [HumanMessage(content="test")]}
        result = search_policy(state)

        assert result["search_results"] == []

    @patch("src.graph.planner.web_search_fn")
    def test_query_contains_current_year(self, mock_search):
        mock_search.return_value = []
        from datetime import datetime

        state = {"messages": [HumanMessage(content="test")]}
        search_policy(state)

        call_args = mock_search.call_args[0][0]
        assert str(datetime.now().year) in call_args


class TestGeneratePlan:

    @patch("src.graph.planner._get_llm")
    def test_generates_plan_message(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("## 复习计划\n- 周一：数学\n- 周二：语文")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="帮我制定下周复习计划")],
            "search_results": [
                {"content": "高考6月7日", "title": "政策", "url": "https://example.com"},
            ],
        }
        result = generate_plan(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "复习计划" in result["messages"][0].content

    @patch("src.graph.planner._get_llm")
    def test_handles_empty_search_results(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("基于通用经验的计划...")
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [HumanMessage(content="帮我做计划")],
            "search_results": [],
        }
        result = generate_plan(state)

        assert len(result["messages"]) == 1
        # Verify the prompt was called with the fallback text
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content  # HumanMessage content
        assert "未获取到最新政策信息" in prompt_text
