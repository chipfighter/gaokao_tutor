"""Unit tests for the Emotional Response node."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.emotional import emotional_response


class TestEmotionalResponse:

    @patch("src.graph.emotional.get_node_llm")
    def test_returns_ai_message(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("同学你好，感到焦虑是很正常的...")
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="我好焦虑")]}
        result = emotional_response(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0

    @patch("src.graph.emotional.get_node_llm")
    def test_passes_full_history(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("response")
        mock_get_llm.return_value = mock_llm

        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！"),
            HumanMessage(content="我压力好大"),
        ]
        state = {"messages": msgs}
        emotional_response(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # First message is SystemMessage (prompt), then the 3 history messages
        assert len(call_args) == 4

    @patch("src.graph.emotional.get_node_llm")
    def test_system_prompt_included(self, mock_get_llm, mock_llm_response):
        from langchain_core.messages import SystemMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("response")
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        emotional_response(state)

        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert "班主任" in call_args[0].content
