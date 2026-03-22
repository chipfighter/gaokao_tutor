"""Unit tests for LLM fallback/resilience mechanism.

Tests cover: invoke_with_fallback helper, per-node fallback behavior,
and tracing attributes on failover events.
All tests mock LLM invocations — no real API calls required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# ===========================================================================
# TestInvokeWithFallback — core helper
# ===========================================================================

class TestInvokeWithFallback:
    """Unit tests for invoke_with_fallback()."""

    def test_returns_primary_response_on_success(self):
        """Primary succeeds → return its response directly."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.return_value = "primary-ok"

        result = invoke_with_fallback(primary, ["msg"])
        assert result == "primary-ok"

    def test_sets_fallback_used_false_on_success(self):
        """Span should record llm.fallback_used=False when primary succeeds."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], span=span)
        span.set_attribute.assert_called_with("llm.fallback_used", False)

    def test_falls_back_on_timeout_error(self):
        """TimeoutError on primary → fallback response returned."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        result = invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    def test_falls_back_on_connection_error(self):
        """ConnectionError on primary → fallback response returned."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("refused")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        result = invoke_with_fallback(primary, ["msg"], fallback=fallback)
        assert result == "fallback-ok"

    def test_sets_fallback_used_true_on_failover(self):
        """Span should record llm.fallback_used=True on failover."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)
        span.set_attribute.assert_any_call("llm.fallback_used", True)

    def test_records_fallback_model_on_span(self):
        """Span should record the fallback model name."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.model_name = "deepseek-lite"
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)
        span.set_attribute.assert_any_call("llm.fallback_model", "deepseek-lite")

    def test_records_fallback_event_on_span(self):
        """Span should have an llm.fallback_triggered event with error details."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.model_name = "deepseek-lite"
        fallback.invoke.return_value = "ok"
        span = MagicMock()

        invoke_with_fallback(primary, ["msg"], fallback=fallback, span=span)

        span.add_event.assert_called_once()
        call_args = span.add_event.call_args
        assert call_args[0][0] == "llm.fallback_triggered"
        assert call_args[0][1]["error_type"] == "TimeoutError"

    def test_propagates_error_when_no_fallback(self):
        """No fallback configured + primary fails → error propagates."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")

        with pytest.raises(TimeoutError):
            invoke_with_fallback(primary, ["msg"])

    def test_propagates_fallback_error(self):
        """Both primary and fallback fail → fallback error propagates."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("primary failed")
        fallback = MagicMock()
        fallback.invoke.side_effect = ConnectionError("fallback also failed")

        with pytest.raises(ConnectionError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

    def test_does_not_catch_value_error(self):
        """Non-recoverable errors (ValueError) should NOT trigger fallback."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = ValueError("bad input")
        fallback = MagicMock()

        with pytest.raises(ValueError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

        fallback.invoke.assert_not_called()

    def test_does_not_catch_key_error(self):
        """KeyError is a programming bug, not an API error — should propagate."""
        from src.graph.llm import invoke_with_fallback

        primary = MagicMock()
        primary.invoke.side_effect = KeyError("missing")
        fallback = MagicMock()

        with pytest.raises(KeyError):
            invoke_with_fallback(primary, ["msg"], fallback=fallback)

        fallback.invoke.assert_not_called()


# ===========================================================================
# TestGenerateAnswerFallback — academic node
# ===========================================================================

class TestGenerateAnswerFallback:
    """Test that generate_answer falls back on primary LLM failure."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_uses_fallback_on_primary_timeout(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("primary timed out")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback.invoke.return_value = mock_llm_response("fallback answer")
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        result = generate_answer(state)

        assert "fallback answer" in result["messages"][0].content
        fallback.invoke.assert_called_once()

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_returns_primary_when_healthy(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.return_value = mock_llm_response("primary answer")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="判别式怎么用")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        result = generate_answer(state)

        assert "primary answer" in result["messages"][0].content
        fallback.invoke.assert_not_called()


# ===========================================================================
# TestGeneratePlanFallback — planner node
# ===========================================================================

class TestGeneratePlanFallback:
    """Test that generate_plan falls back on primary LLM failure."""

    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner._get_llm")
    def test_uses_fallback_on_primary_timeout(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("primary timed out")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback.invoke.return_value = mock_llm_response("fallback plan")
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="帮我制定复习计划")],
            "search_results": [],
        }

        from src.graph.planner import generate_plan

        result = generate_plan(state)

        assert "fallback plan" in result["messages"][0].content
        fallback.invoke.assert_called_once()

    @patch("src.graph.planner.get_fallback_llm")
    @patch("src.graph.planner._get_llm")
    def test_returns_primary_when_healthy(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.return_value = mock_llm_response("primary plan")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="帮我做计划")],
            "search_results": [],
        }

        from src.graph.planner import generate_plan

        result = generate_plan(state)

        assert "primary plan" in result["messages"][0].content
        fallback.invoke.assert_not_called()


# ===========================================================================
# TestEmotionalResponseFallback — emotional node
# ===========================================================================

class TestEmotionalResponseFallback:
    """Test that emotional_response falls back on primary LLM failure."""

    @patch("src.graph.emotional.get_fallback_llm")
    @patch("src.graph.emotional._get_llm")
    def test_uses_fallback_on_primary_timeout(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("primary timed out")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback.invoke.return_value = mock_llm_response("fallback comfort")
        mock_get_fallback.return_value = fallback

        state = {"messages": [HumanMessage(content="我好焦虑")]}

        from src.graph.emotional import emotional_response

        result = emotional_response(state)

        assert "fallback comfort" in result["messages"][0].content
        fallback.invoke.assert_called_once()

    @patch("src.graph.emotional.get_fallback_llm")
    @patch("src.graph.emotional._get_llm")
    def test_returns_primary_when_healthy(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        primary = MagicMock()
        primary.invoke.return_value = mock_llm_response("primary comfort")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        mock_get_fallback.return_value = fallback

        state = {"messages": [HumanMessage(content="我好焦虑")]}

        from src.graph.emotional import emotional_response

        result = emotional_response(state)

        assert "primary comfort" in result["messages"][0].content
        fallback.invoke.assert_not_called()


# ===========================================================================
# TestFallbackTracing — OTel span attributes on failover
# ===========================================================================

class TestFallbackTracing:
    """Test that fallback events appear correctly in OTel spans."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_span_records_fallback_attributes(
        self, mock_get_llm, mock_get_fallback,
        in_memory_exporter, mock_llm_response,
    ):
        """When fallback triggers, the llm.invoke span must record it."""
        primary = MagicMock()
        primary.invoke.side_effect = TimeoutError("timed out")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback.model_name = "deepseek-lite"
        fallback.invoke.return_value = mock_llm_response("traced fallback")
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        generate_answer(state)

        spans = in_memory_exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name.startswith("llm.invoke")]
        assert len(llm_spans) >= 1

        llm_span = llm_spans[0]
        attrs = dict(llm_span.attributes)
        assert attrs["llm.fallback_used"] is True
        assert attrs["llm.fallback_model"] == "deepseek-lite"

        # Should have a fallback_triggered event
        event_names = [e.name for e in llm_span.events]
        assert "llm.fallback_triggered" in event_names

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic._get_llm")
    def test_span_records_no_fallback_on_success(
        self, mock_get_llm, mock_get_fallback,
        in_memory_exporter, mock_llm_response,
    ):
        """When primary succeeds, span should record llm.fallback_used=False."""
        primary = MagicMock()
        primary.invoke.return_value = mock_llm_response("primary ok")
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="test")],
            "context": [],
        }

        from src.graph.academic import generate_answer

        generate_answer(state)

        spans = in_memory_exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.name.startswith("llm.invoke")]
        assert len(llm_spans) >= 1

        attrs = dict(llm_spans[0].attributes)
        assert attrs["llm.fallback_used"] is False
