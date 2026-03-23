"""Unit tests for hallucination evaluation and retry loop.

Tests cover: HallucinationEvaluation schema, evaluate_hallucination node,
should_retry_or_end conditional edge, OTel tracing, fallback resilience,
and generate_answer compatibility with retry loops.
All tests mock LLM invocations -- no real API calls required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.academic import (
    MAX_RETRIES,
    HallucinationEvaluation,
    evaluate_hallucination,
    should_retry_or_end,
)


# ===========================================================================
# TestHallucinationEvalSchema -- Pydantic model validation
# ===========================================================================

class TestHallucinationEvalSchema:
    """Validate the structured output Pydantic model."""

    def test_faithful_evaluation(self):
        e = HallucinationEvaluation(
            is_faithful=True, reason="Answer is grounded in context",
        )
        assert e.is_faithful is True
        assert e.reason == "Answer is grounded in context"

    def test_unfaithful_evaluation(self):
        e = HallucinationEvaluation(
            is_faithful=False, reason="Fabricated formula not in context",
        )
        assert e.is_faithful is False
        assert "Fabricated" in e.reason


# ===========================================================================
# TestEvaluateHallucinationNode -- core node logic
# ===========================================================================

class TestEvaluateHallucinationNode:
    """Test the evaluate_hallucination graph node."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_faithful_answer_not_flagged(self, mock_get_llm, mock_get_fallback):
        """Faithful answer -> hallucination_detected=False, no retry_count change."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=True, reason="Good",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="question"), AIMessage(content="answer")],
            "context": [{"content": "context"}],
            "retry_count": 0,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is False
        assert "retry_count" not in result

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_unfaithful_answer_detected(self, mock_get_llm, mock_get_fallback):
        """Hallucinating answer -> hallucination_detected=True, retry_count incremented."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=False, reason="Fabricated",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="bad")],
            "context": [],
            "retry_count": 0,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is True
        assert result["retry_count"] == 1

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_increments_retry_count(self, mock_get_llm, mock_get_fallback):
        """retry_count=1 + hallucination -> retry_count=2."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=False, reason="Off-topic",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="bad")],
            "context": [],
            "retry_count": 1,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is True
        assert result["retry_count"] == 2

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_defaults_to_valid_on_parse_failure(self, mock_get_llm, mock_get_fallback):
        """Structured output parsing fails -> default to valid (don't block answer)."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("OutputParserException")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "context": [],
            "retry_count": 0,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is False

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_extracts_last_human_message_as_question(
        self, mock_get_llm, mock_get_fallback,
    ):
        """Should use the last HumanMessage as the question, not the last message."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=True, reason="OK",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [
                HumanMessage(content="the real question"),
                AIMessage(content="first attempt"),
                AIMessage(content="second attempt"),
            ],
            "context": [],
            "retry_count": 1,
        }

        evaluate_hallucination(state)

        # The eval prompt sent to the structured LLM should contain the question
        call_args = mock_structured.invoke.call_args[0][0]
        prompt_text = call_args[-1].content
        assert "the real question" in prompt_text


# ===========================================================================
# TestShouldRetryOrEnd -- conditional edge routing
# ===========================================================================

class TestShouldRetryOrEnd:
    """Test the conditional edge function for retry/end routing."""

    def test_routes_end_when_valid(self):
        state = {"hallucination_detected": False, "retry_count": 0}
        assert should_retry_or_end(state) == "end"

    def test_routes_retry_first_attempt(self):
        """retry_count=1 (first retry) with hallucination -> retry."""
        state = {"hallucination_detected": True, "retry_count": 1}
        assert should_retry_or_end(state) == "retry"

    def test_routes_retry_at_max(self):
        """retry_count=MAX_RETRIES -> still allows this retry."""
        state = {"hallucination_detected": True, "retry_count": MAX_RETRIES}
        assert should_retry_or_end(state) == "retry"

    def test_routes_end_past_max(self):
        """retry_count > MAX_RETRIES -> stop retrying."""
        state = {"hallucination_detected": True, "retry_count": MAX_RETRIES + 1}
        assert should_retry_or_end(state) == "end"

    def test_defaults_end_when_no_flag(self):
        """Missing hallucination_detected defaults to end."""
        state = {"retry_count": 0}
        assert should_retry_or_end(state) == "end"

    def test_defaults_end_when_empty_state(self):
        state = {}
        assert should_retry_or_end(state) == "end"


# ===========================================================================
# TestHallucinationTracing -- OTel span attributes
# ===========================================================================

class TestHallucinationTracing:
    """Verify OTel spans record hallucination metadata."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_span_records_hallucination_detected(
        self, mock_get_llm, mock_get_fallback, in_memory_exporter,
    ):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=False, reason="Bad",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "context": [],
            "retry_count": 0,
        }

        evaluate_hallucination(state)

        spans = in_memory_exporter.get_finished_spans()
        node_spans = [s for s in spans if s.name == "graph.node.evaluate_hallucination"]
        assert len(node_spans) == 1
        attrs = dict(node_spans[0].attributes)
        assert attrs["graph.node.hallucination_detected"] is True

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_span_records_retry_count(
        self, mock_get_llm, mock_get_fallback, in_memory_exporter,
    ):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=False, reason="Bad",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_fallback.with_structured_output.return_value = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "context": [],
            "retry_count": 1,
        }

        evaluate_hallucination(state)

        spans = in_memory_exporter.get_finished_spans()
        node_spans = [s for s in spans if s.name == "graph.node.evaluate_hallucination"]
        assert len(node_spans) == 1
        attrs = dict(node_spans[0].attributes)
        assert attrs["graph.node.retry_count"] == 2


# ===========================================================================
# TestEvaluateHallucinationFallback -- fallback resilience
# ===========================================================================

class TestEvaluateHallucinationFallback:
    """Test that evaluate_hallucination uses fallback on primary LLM failure."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_uses_fallback_on_primary_timeout(self, mock_get_llm, mock_get_fallback):
        primary = MagicMock()
        primary_structured = MagicMock()
        primary_structured.invoke.side_effect = TimeoutError("timed out")
        primary.with_structured_output.return_value = primary_structured
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback_structured = MagicMock()
        fallback_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=True, reason="Fallback OK",
        )
        fallback.with_structured_output.return_value = fallback_structured
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "context": [],
            "retry_count": 0,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is False
        fallback_structured.invoke.assert_called_once()

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_returns_primary_when_healthy(self, mock_get_llm, mock_get_fallback):
        primary = MagicMock()
        primary_structured = MagicMock()
        primary_structured.invoke.return_value = HallucinationEvaluation(
            is_faithful=True, reason="Primary OK",
        )
        primary.with_structured_output.return_value = primary_structured
        mock_get_llm.return_value = primary

        fallback = MagicMock()
        fallback_structured = MagicMock()
        fallback.with_structured_output.return_value = fallback_structured
        mock_get_fallback.return_value = fallback

        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "context": [],
            "retry_count": 0,
        }

        result = evaluate_hallucination(state)

        assert result["hallucination_detected"] is False
        fallback_structured.invoke.assert_not_called()


# ===========================================================================
# TestGenerateAnswerRetryCompat -- generate_answer works in retry loops
# ===========================================================================

class TestGenerateAnswerRetryCompat:
    """generate_answer must use the original HumanMessage during retry loops."""

    @patch("src.graph.academic.get_fallback_llm")
    @patch("src.graph.academic.get_node_llm")
    def test_uses_last_human_message_not_last_message(
        self, mock_get_llm, mock_get_fallback, mock_llm_response,
    ):
        """On retry, messages include previous AI answers.
        generate_answer should find the user's original question."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_llm_response("new answer")
        mock_get_llm.return_value = mock_llm

        mock_fallback = MagicMock()
        mock_get_fallback.return_value = mock_fallback

        state = {
            "messages": [
                HumanMessage(content="What is the discriminant?"),
                AIMessage(content="Previous bad answer"),
            ],
            "context": [],
        }

        from src.graph.academic import generate_answer

        generate_answer(state)

        # Verify the prompt contains the human question, not the AI answer
        call_args = mock_llm.invoke.call_args[0][0]
        human_msgs = [m for m in call_args if isinstance(m, HumanMessage)]
        prompt_text = " ".join(m.content for m in human_msgs)
        assert "What is the discriminant?" in prompt_text
        assert "Previous bad answer" not in prompt_text
