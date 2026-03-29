"""Unit tests for the Supervisor node (intent routing + keypoint extraction)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.graph.supervisor import _VALID_INTENTS, route_by_intent, supervisor_node


class TestSupervisorNode:

    @patch("src.graph.supervisor.get_node_llm")
    async def test_academic_intent(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response(
            json.dumps({"intent": "academic", "subject": "math", "keypoints": ["二次函数", "判别式"]})
        ))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="二次函数的判别式怎么用？")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "math"
        assert "判别式" in result["keypoints"]

    @patch("src.graph.supervisor.get_node_llm")
    async def test_planning_intent(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response(
            json.dumps({"intent": "planning", "subject": "other", "keypoints": []})
        ))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="帮我制定复习计划")]}
        result = await supervisor_node(state)

        assert result["intent"] == "planning"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_emotional_intent(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response(
            json.dumps({"intent": "emotional", "subject": "other", "keypoints": []})
        ))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="我好焦虑")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_invalid_intent_falls_back_to_academic(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response(
            json.dumps({"intent": "invalid_intent", "subject": "other", "keypoints": []})
        ))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="随便说点什么")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"

    @patch("src.graph.supervisor.get_node_llm")
    async def test_malformed_json_falls_back(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response("not valid json {{{"))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="test")]}
        result = await supervisor_node(state)

        assert result["intent"] == "academic"
        assert result["subject"] == "other"
        assert result["keypoints"] == []

    @patch("src.graph.supervisor.get_node_llm")
    async def test_missing_fields_use_defaults(self, mock_get_llm, mock_llm_response):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response(json.dumps({"intent": "emotional"})))
        mock_get_llm.return_value = mock_llm

        state = {"messages": [HumanMessage(content="你好")]}
        result = await supervisor_node(state)

        assert result["intent"] == "emotional"
        assert result["subject"] == "other"
        assert result["keypoints"] == []


class TestRouteByIntent:

    def test_routes_academic(self):
        assert route_by_intent({"intent": "academic"}) == "academic"

    def test_routes_planning(self):
        assert route_by_intent({"intent": "planning"}) == "planning"

    def test_routes_emotional(self):
        assert route_by_intent({"intent": "emotional"}) == "emotional"

    def test_missing_intent_defaults_to_academic(self):
        assert route_by_intent({}) == "academic"


class TestValidIntents:

    def test_valid_intents_set(self):
        assert _VALID_INTENTS == {"academic", "planning", "emotional"}
