"""Unit tests for the OpenTelemetry tracing module.

Tests cover: SQLite exporter, collector setup, @traced_node decorator,
and context managers (traced_llm_call, traced_retrieval, traced_search).
All tests are offline — no Jaeger or real exporters required.
"""

from __future__ import annotations

import os
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_mock_span(
    trace_id: int = 0x1234567890ABCDEF1234567890ABCDEF,
    span_id: int = 0x1234567890ABCDEF,
    parent_span_id: int | None = None,
    name: str = "test_span",
    attributes: dict | None = None,
) -> MagicMock:
    """Create a mock ReadableSpan with all required properties."""
    span = MagicMock(spec=ReadableSpan)

    ctx = MagicMock()
    ctx.trace_id = trace_id
    ctx.span_id = span_id
    span.context = ctx

    if parent_span_id is not None:
        parent = MagicMock()
        parent.span_id = parent_span_id
        span.parent = parent
    else:
        span.parent = None

    span.name = name
    span.kind = SpanKind.INTERNAL
    span.start_time = 1_000_000_000  # 1 second in nanoseconds
    span.end_time = 2_000_000_000
    span.status = MagicMock()
    span.status.status_code = StatusCode.OK
    span.attributes = attributes or {"key": "value"}
    span.events = []
    span.resource = Resource.create({"service.name": "test"})
    return span


# ===========================================================================
# TestSQLiteSpanExporter
# ===========================================================================

class TestSQLiteSpanExporter:
    """Tests for the custom SQLite fallback exporter."""

    def test_creates_table_on_init(self, tmp_path):
        """Table 'spans' should exist after initialization."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        exporter.shutdown()

    def test_exports_spans_successfully(self, tmp_path):
        """export() should insert span rows and return SUCCESS."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)

        spans = [_create_mock_span(name="span_1"), _create_mock_span(name="span_2")]
        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        assert count == 2
        conn.close()
        exporter.shutdown()

    def test_stores_correct_trace_id(self, tmp_path):
        """Trace IDs should be stored as 32-char hex strings."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)

        span = _create_mock_span(trace_id=0xAABBCCDDEEFF00112233445566778899)
        exporter.export([span])

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT trace_id FROM spans").fetchone()
        assert row[0] == "aabbccddeeff00112233445566778899"
        assert len(row[0]) == 32
        conn.close()
        exporter.shutdown()

    def test_stores_attributes_as_json(self, tmp_path):
        """Span attributes should be JSON-serializable in the DB."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter
        import json

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)

        span = _create_mock_span(attributes={"model": "deepseek", "temp": 0.7})
        exporter.export([span])

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT attributes FROM spans").fetchone()
        attrs = json.loads(row[0])
        assert attrs["model"] == "deepseek"
        assert attrs["temp"] == 0.7
        conn.close()
        exporter.shutdown()

    def test_handles_empty_spans_list(self, tmp_path):
        """export([]) should succeed without inserting rows."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)

        result = exporter.export([])
        assert result == SpanExportResult.SUCCESS

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        assert count == 0
        conn.close()
        exporter.shutdown()

    def test_returns_failure_on_db_error(self, tmp_path):
        """If the DB write fails, export() should return FAILURE."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)
        # Close internal connection to force a write error
        exporter._conn.close()

        result = exporter.export([_create_mock_span()])
        assert result == SpanExportResult.FAILURE

    def test_shutdown_closes_connection(self, tmp_path):
        """After shutdown(), the connection should be closed."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "test.db")
        exporter = SQLiteSpanExporter(db_path)
        exporter.shutdown()

        # Attempting to use the closed connection should fail
        result = exporter.export([_create_mock_span()])
        assert result == SpanExportResult.FAILURE

    def test_creates_parent_directory(self, tmp_path):
        """Exporter should create parent directories if they don't exist."""
        from src.tracing.sqlite_exporter import SQLiteSpanExporter

        db_path = str(tmp_path / "nested" / "dir" / "traces.db")
        exporter = SQLiteSpanExporter(db_path)

        assert os.path.exists(db_path)
        exporter.shutdown()


# ===========================================================================
# TestSetupTracing
# ===========================================================================

class TestSetupTracing:
    """Tests for the collector setup_tracing() function."""

    @staticmethod
    def _reset():
        """Force-reset OTel global state and collector module state."""
        from opentelemetry.util._once import Once
        trace._TRACER_PROVIDER_SET_ONCE = Once()
        trace._TRACER_PROVIDER = None
        import src.tracing.collector as col
        col._tracer_provider = None

    def setup_method(self):
        """Reset global state before each test."""
        self._reset()

    def teardown_method(self):
        """Clean up after each test."""
        import src.tracing.collector as col
        if col._tracer_provider is not None:
            col._tracer_provider.shutdown()
        self._reset()

    @patch.dict(os.environ, {"OTEL_TRACING_ENABLED": "false"})
    def test_disabled_returns_none(self):
        """When OTEL_TRACING_ENABLED=false, setup_tracing() returns None."""
        from src.tracing.collector import setup_tracing

        result = setup_tracing()
        assert result is None

    @patch.dict(os.environ, {
        "OTEL_TRACING_ENABLED": "true",
        "OTEL_TRACES_EXPORTER": "none",
    })
    def test_none_exporter_creates_provider(self):
        """Exporter 'none' should still create a TracerProvider."""
        from src.tracing.collector import setup_tracing

        provider = setup_tracing()
        assert provider is not None
        assert isinstance(provider, TracerProvider)

    @patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter")
    @patch.dict(os.environ, {
        "OTEL_TRACING_ENABLED": "true",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
    })
    def test_otlp_exporter_configured(self, mock_otlp_cls, tmp_path):
        """When exporter is 'otlp', OTLPSpanExporter should be created."""
        from src.tracing.collector import setup_tracing

        with patch.dict(os.environ, {"OTEL_SQLITE_FALLBACK_PATH": str(tmp_path / "t.db")}):
            provider = setup_tracing()

        assert provider is not None
        mock_otlp_cls.assert_called_once()

    @patch.dict(os.environ, {
        "OTEL_TRACING_ENABLED": "true",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_SERVICE_NAME": "test-service",
    })
    def test_service_name_from_env(self):
        """Resource should use OTEL_SERVICE_NAME from environment."""
        from src.tracing.collector import setup_tracing

        provider = setup_tracing()
        resource = provider.resource
        assert resource.attributes.get("service.name") == "test-service"

    def test_get_tracer_returns_tracer(self):
        """get_tracer() should return a Tracer instance (even no-op)."""
        from src.tracing.collector import get_tracer

        tracer = get_tracer()
        assert tracer is not None

    @patch.dict(os.environ, {
        "OTEL_TRACING_ENABLED": "true",
        "OTEL_TRACES_EXPORTER": "sqlite",
    })
    def test_sqlite_only_exporter(self, tmp_path):
        """When exporter is 'sqlite', only SQLite exporter should be used."""
        from src.tracing.collector import setup_tracing

        with patch.dict(os.environ, {"OTEL_SQLITE_FALLBACK_PATH": str(tmp_path / "t.db")}):
            provider = setup_tracing()

        assert provider is not None
        assert isinstance(provider, TracerProvider)


# ===========================================================================
# TestTracedNodeDecorator
# ===========================================================================

class TestTracedNodeDecorator:
    """Tests for the @traced_node decorator."""

    def test_preserves_function_name(self, in_memory_exporter):
        """The wrapper should preserve __name__ via functools.wraps."""
        from src.tracing.decorators import traced_node

        @traced_node
        def my_node(state):
            return {"intent": "academic"}

        assert my_node.__name__ == "my_node"

    def test_returns_original_result(self, in_memory_exporter):
        """The decorator should not alter the return value."""
        from src.tracing.decorators import traced_node

        @traced_node
        def my_node(state):
            return {"intent": "academic", "subject": "math"}

        result = my_node({"messages": []})
        assert result == {"intent": "academic", "subject": "math"}

    def test_creates_span_with_node_name(self, in_memory_exporter):
        """A span named 'graph.node.<func_name>' should be created."""
        from src.tracing.decorators import traced_node

        @traced_node
        def supervisor_node(state):
            return {"intent": "academic"}

        supervisor_node({"messages": []})

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "graph.node.supervisor_node"

    def test_records_input_keys(self, in_memory_exporter):
        """Span should have attribute 'graph.node.input_keys'."""
        from src.tracing.decorators import traced_node

        @traced_node
        def my_node(state):
            return {}

        my_node({"messages": [], "intent": "academic"})

        spans = in_memory_exporter.get_finished_spans()
        input_keys = spans[0].attributes["graph.node.input_keys"]
        assert "messages" in input_keys
        assert "intent" in input_keys

    def test_records_output_keys(self, in_memory_exporter):
        """Span should have attribute 'graph.node.output_keys'."""
        from src.tracing.decorators import traced_node

        @traced_node
        def my_node(state):
            return {"intent": "academic", "subject": "math"}

        my_node({"messages": []})

        spans = in_memory_exporter.get_finished_spans()
        output_keys = spans[0].attributes["graph.node.output_keys"]
        assert "intent" in output_keys
        assert "subject" in output_keys

    def test_records_intent_when_present(self, in_memory_exporter):
        """If the node returns 'intent', it should be a span attribute."""
        from src.tracing.decorators import traced_node

        @traced_node
        def my_node(state):
            return {"intent": "emotional"}

        my_node({"messages": []})

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["graph.node.intent"] == "emotional"

    def test_records_context_count_when_present(self, in_memory_exporter):
        """If the node returns 'context', count should be recorded."""
        from src.tracing.decorators import traced_node

        @traced_node
        def rag_node(state):
            return {"context": [{"type": "rag", "content": "a"}, {"type": "rag", "content": "b"}]}

        rag_node({"messages": []})

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["graph.node.context_count"] == 2

    def test_records_search_result_count(self, in_memory_exporter):
        """If the node returns 'search_results', count should be recorded."""
        from src.tracing.decorators import traced_node

        @traced_node
        def search_node(state):
            return {"search_results": [{"title": "r1"}]}

        search_node({"messages": []})

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["graph.node.search_result_count"] == 1

    def test_reraises_exception(self, in_memory_exporter):
        """Exceptions should propagate after being recorded on the span."""
        from src.tracing.decorators import traced_node

        @traced_node
        def failing_node(state):
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_node({})

    def test_records_exception_on_span(self, in_memory_exporter):
        """On exception, span status should be ERROR and exception recorded."""
        from src.tracing.decorators import traced_node

        @traced_node
        def failing_node(state):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            failing_node({})

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        # Exception event should be recorded
        assert len(spans[0].events) > 0
        assert spans[0].events[0].name == "exception"


# ===========================================================================
# TestTracedLlmCall
# ===========================================================================

class TestTracedLlmCall:
    """Tests for the traced_llm_call context manager."""

    def test_creates_llm_span(self, in_memory_exporter):
        """A span named 'llm.invoke.<node_name>' should be created."""
        from src.tracing.decorators import traced_llm_call

        with traced_llm_call(model_name="deepseek-chat", node_name="supervisor"):
            pass  # simulate LLM call

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "llm.invoke.supervisor"

    def test_records_model_name(self, in_memory_exporter):
        """Span should have attribute 'llm.model'."""
        from src.tracing.decorators import traced_llm_call

        with traced_llm_call(model_name="deepseek-chat", node_name="test"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["llm.model"] == "deepseek-chat"

    def test_records_temperature(self, in_memory_exporter):
        """Span should have attribute 'llm.temperature' when provided."""
        from src.tracing.decorators import traced_llm_call

        with traced_llm_call(model_name="m", node_name="n", temperature=0.7):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["llm.temperature"] == 0.7

    def test_records_latency(self, in_memory_exporter):
        """Span should have attribute 'llm.latency_ms'."""
        from src.tracing.decorators import traced_llm_call

        with traced_llm_call(model_name="m", node_name="n") as span:
            time.sleep(0.01)  # Small delay so latency > 0

        spans = in_memory_exporter.get_finished_spans()
        latency = spans[0].attributes["llm.latency_ms"]
        assert latency > 0

    def test_yields_span_for_custom_attributes(self, in_memory_exporter):
        """Caller should be able to set custom attributes on the yielded span."""
        from src.tracing.decorators import traced_llm_call

        with traced_llm_call(model_name="m", node_name="n") as span:
            span.set_attribute("llm.token_count.input", 50)
            span.set_attribute("llm.token_count.output", 120)

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["llm.token_count.input"] == 50
        assert spans[0].attributes["llm.token_count.output"] == 120

    def test_records_exception(self, in_memory_exporter):
        """On error, span should record the exception and set ERROR status."""
        from src.tracing.decorators import traced_llm_call

        with pytest.raises(RuntimeError):
            with traced_llm_call(model_name="m", node_name="n"):
                raise RuntimeError("API timeout")

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
        assert len(spans[0].events) > 0


# ===========================================================================
# TestTracedRetrieval
# ===========================================================================

class TestTracedRetrieval:
    """Tests for the traced_retrieval context manager."""

    def test_creates_rag_span(self, in_memory_exporter):
        """A span named 'rag.retrieve' should be created."""
        from src.tracing.decorators import traced_retrieval

        with traced_retrieval(query="quadratic formula"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "rag.retrieve"

    def test_records_query(self, in_memory_exporter):
        """Span should have attribute 'rag.query'."""
        from src.tracing.decorators import traced_retrieval

        with traced_retrieval(query="discriminant usage"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["rag.query"] == "discriminant usage"

    def test_truncates_long_query(self, in_memory_exporter):
        """Long queries should be truncated to 200 chars."""
        from src.tracing.decorators import traced_retrieval

        long_query = "x" * 300
        with traced_retrieval(query=long_query):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans[0].attributes["rag.query"]) == 200

    def test_records_subject(self, in_memory_exporter):
        """Span should have attribute 'rag.subject'."""
        from src.tracing.decorators import traced_retrieval

        with traced_retrieval(query="q", subject="math"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["rag.subject"] == "math"

    def test_default_subject_is_all(self, in_memory_exporter):
        """When subject is None, attribute should be 'all'."""
        from src.tracing.decorators import traced_retrieval

        with traced_retrieval(query="q"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["rag.subject"] == "all"

    def test_caller_can_set_result_attributes(self, in_memory_exporter):
        """Caller should be able to set doc_count, is_hit, top_score."""
        from src.tracing.decorators import traced_retrieval

        with traced_retrieval(query="q") as span:
            span.set_attribute("rag.doc_count", 3)
            span.set_attribute("rag.is_hit", True)
            span.set_attribute("rag.top_score", 0.85)

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["rag.doc_count"] == 3
        assert spans[0].attributes["rag.is_hit"] is True
        assert spans[0].attributes["rag.top_score"] == 0.85


# ===========================================================================
# TestTracedSearch
# ===========================================================================

class TestTracedSearch:
    """Tests for the traced_search context manager."""

    def test_creates_search_span(self, in_memory_exporter):
        """A span named 'web.search' should be created."""
        from src.tracing.decorators import traced_search

        with traced_search(query="gaokao 2026 policy"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "web.search"

    def test_records_query(self, in_memory_exporter):
        """Span should have attribute 'search.query'."""
        from src.tracing.decorators import traced_search

        with traced_search(query="gaokao reform"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["search.query"] == "gaokao reform"

    def test_records_timeout(self, in_memory_exporter):
        """Span should have attribute 'search.timeout_sec'."""
        from src.tracing.decorators import traced_search

        with traced_search(query="q", timeout=30):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["search.timeout_sec"] == 30

    def test_default_timeout_is_15(self, in_memory_exporter):
        """Default timeout should be 15 seconds."""
        from src.tracing.decorators import traced_search

        with traced_search(query="q"):
            pass

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["search.timeout_sec"] == 15

    def test_caller_can_set_result_attributes(self, in_memory_exporter):
        """Caller should be able to set result_count and timed_out."""
        from src.tracing.decorators import traced_search

        with traced_search(query="q") as span:
            span.set_attribute("search.result_count", 5)
            span.set_attribute("search.timed_out", False)

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].attributes["search.result_count"] == 5
        assert spans[0].attributes["search.timed_out"] is False

    def test_records_exception(self, in_memory_exporter):
        """On error, span should record the exception."""
        from src.tracing.decorators import traced_search

        with pytest.raises(TimeoutError):
            with traced_search(query="q"):
                raise TimeoutError("search timed out")

        spans = in_memory_exporter.get_finished_spans()
        assert spans[0].status.status_code == StatusCode.ERROR
