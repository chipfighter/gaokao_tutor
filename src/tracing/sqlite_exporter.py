"""SQLite fallback exporter -- persists spans locally when Jaeger is unreachable."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    kind TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    status TEXT,
    attributes TEXT,
    events TEXT,
    resource TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteSpanExporter(SpanExporter):
    """Custom SpanExporter that writes completed spans to a local SQLite database.

    Intended as a fallback when the primary OTLP/Jaeger exporter is unavailable,
    ensuring no trace data is lost during local development.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Insert spans into SQLite within a single transaction."""
        if not spans:
            return SpanExportResult.SUCCESS

        rows = []
        for span in spans:
            try:
                parent_id = (
                    format(span.parent.span_id, "016x") if span.parent else None
                )
                events = [
                    {
                        "name": e.name,
                        "timestamp": str(e.timestamp),
                        "attributes": dict(e.attributes) if e.attributes else {},
                    }
                    for e in (span.events or [])
                ]
                rows.append((
                    format(span.context.trace_id, "032x"),
                    format(span.context.span_id, "016x"),
                    parent_id,
                    span.name,
                    str(span.kind),
                    str(span.start_time),
                    str(span.end_time),
                    str(span.status.status_code),
                    json.dumps(dict(span.attributes) if span.attributes else {}, default=str),
                    json.dumps(events, default=str),
                    json.dumps(
                        dict(span.resource.attributes) if span.resource else {},
                        default=str,
                    ),
                ))
            except Exception:
                logger.exception("Failed to serialize span: %s", span.name)
                continue

        try:
            with self._lock:
                self._conn.executemany(
                    """INSERT INTO spans
                       (trace_id, span_id, parent_span_id, name, kind,
                        start_time, end_time, status, attributes, events, resource)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                self._conn.commit()
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("SQLite span export failed")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Close the SQLite connection."""
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int | None = None) -> bool:
        """No-op for SQLite (writes are synchronous)."""
        return True
