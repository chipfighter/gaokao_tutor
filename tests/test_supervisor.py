"""Unit tests for the Supervisor node (intent routing + keypoint extraction)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.supervisor import (
    SupervisorOutput,
    _VALID_INTENTS,
    handle_unknown,
    route_by_intent,
    supervisor_node,
)


def _mock_supervisor_output(intent="academic", keywords=None, confidence=0.9):
    """Helper to create a SupervisorOutput instance for mocking."""
    return SupervisorOutput(
        intent=intent,
        keywords=keywords or [],
        confidence=confidence,
    )


class TestSupervisorNode:

    @patch("src.graph.supervisor.get_node_llm")
    async def test_academic_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["二次函数", "判别式"], confidence=0.95,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="二次函数的判别式怎么用？")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "math"
        assert "判别式" in result["keypoints"]
        mock_llm.with_structured_output.assert_called_once_with(SupervisorOutput)

    @patch("src.graph.supervisor.get_node_llm")
    async def test_planning_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="planning", keywords=[], confidence=0.9,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="帮我制定复习计划")]}
        result = await supervisor_node(state)

        assert result["intent"] == "planning"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_emotional_intent(self, mock_get_llm):
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="emotional", keywords=[], confidence=0.85,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="我好焦虑")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_unknown_intent(self, mock_get_llm):
        """Unknown intent is returned when the query is off-topic."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="unknown", keywords=[], confidence=0.3,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="今天天气怎么样")]}
        result = await supervisor_node(state)

        assert result["intent"] == "unknown"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_historical_exam_query_routes_to_academic(self, mock_get_llm):
        """Historical exam queries like '2024高考作文题' should route to academic."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["2024高考", "作文题"], confidence=0.95,
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="2024年高考全国卷语文作文题是什么？")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "chinese"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_structured_output_failure_falls_back(self, mock_get_llm):
        """When structured output fails, fall back to academic defaults."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "other"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_uses_with_structured_output(self, mock_get_llm):
        """Verify with_structured_output is called (no json.loads)."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output())
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        await supervisor_node(state)

        mock_llm.with_structured_output.assert_called_once_with(SupervisorOutput)
        structured_llm.ainvoke.assert_called_once()

    @patch("src.graph.supervisor.get_node_llm")
    async def test_keywords_mapped_to_keypoints(self, mock_get_llm):
        """SupervisorOutput.keywords should be mapped to state keypoints."""
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(return_value=_mock_supervisor_output(
            intent="academic", keywords=["椭圆", "离心率"],
        ))
        mock_llm.with_structured_output.return_value = structured_llm
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="椭圆的离心率怎么求？")]}
        result = await supervisor_node(state)

        assert result["keypoints"] == ["椭圆", "离心率"]


class TestRouteByIntent:

    def test_routes_academic(self):
        assert route_by_intent({"intent": "academic"}) == "academic"

    def test_routes_planning(self):
        assert route_by_intent({"intent": "planning"}) == "planning"

    def test_routes_emotional(self):
        assert route_by_intent({"intent": "emotional"}) == "emotional"

    def test_routes_unknown(self):
        assert route_by_intent({"intent": "unknown"}) == "unknown"

    def test_missing_intent_defaults_to_academic(self):
        assert route_by_intent({}) == "academic"


class TestValidIntents:

    def test_valid_intents_includes_unknown(self):
        assert "unknown" in _VALID_INTENTS

    def test_valid_intents_set(self):
        assert _VALID_INTENTS == {"academic", "planning", "emotional", "unknown"}


class TestHandleUnknown:

    async def test_returns_friendly_message(self):
        state = {"messages": [HumanMessage(content="今天天气怎么样")]}
        result = await handle_unknown(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        # Should contain a friendly message in Chinese
        assert len(result["messages"][0].content) > 0

    async def test_message_is_ai_message(self):
        state = {"messages": [HumanMessage(content="帮我订外卖")]}
        result = await handle_unknown(state)

        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)


class TestSupervisorOutput:

    def test_valid_output(self):
        output = SupervisorOutput(intent="academic", keywords=["数学"], confidence=0.9)
        assert output.intent == "academic"
        assert output.keywords == ["数学"]
        assert output.confidence == 0.9

    def test_unknown_intent_valid(self):
        output = SupervisorOutput(intent="unknown", keywords=[], confidence=0.1)
        assert output.intent == "unknown"

    def test_invalid_intent_raises(self):
        with pytest.raises(Exception):
            SupervisorOutput(intent="invalid", keywords=[], confidence=0.5)
