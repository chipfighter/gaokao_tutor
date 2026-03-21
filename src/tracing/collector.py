"""OpenTelemetry collector -- TracerProvider setup with OTLP + SQLite fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def setup_tracing() -> TracerProvider | None:
    """Initialize the OpenTelemetry TracerProvider with configured exporters.

    Reads configuration from environment variables:
        OTEL_TRACING_ENABLED  -- "true"/"false" kill switch (default "true")
        OTEL_SERVICE_NAME     -- resource service name (default "gaokao-tutor")
        OTEL_TRACES_EXPORTER  -- "otlp", "sqlite", or "none" (default "otlp")
        OTEL_EXPORTER_OTLP_ENDPOINT -- gRPC endpoint (default "localhost:4317")
        OTEL_SQLITE_FALLBACK_PATH   -- SQLite DB path (default "logs/traces.db")

    Returns:
        The configured TracerProvider, or None if tracing is disabled.
    """
    global _tracer_provider

    enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower()
    if enabled != "true":
        logger.info("OpenTelemetry tracing is disabled (OTEL_TRACING_ENABLED=%s)", enabled)
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "gaokao-tutor")
    exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "otlp").lower()

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Primary exporter: OTLP to Jaeger
    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP exporter configured -> %s", endpoint)
        except Exception:
            logger.exception("Failed to configure OTLP exporter, continuing with fallback only")

    # SQLite fallback (always added unless exporter is "none")
    if exporter_type != "none":
        try:
            from src.tracing.sqlite_exporter import SQLiteSpanExporter

            db_path = os.getenv("OTEL_SQLITE_FALLBACK_PATH", "logs/traces.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            sqlite_exporter = SQLiteSpanExporter(db_path)
            provider.add_span_processor(BatchSpanProcessor(sqlite_exporter))
            logger.info("SQLite fallback exporter configured -> %s", db_path)
        except Exception:
            logger.exception("Failed to configure SQLite fallback exporter")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    logger.info(
        "OpenTelemetry tracing initialized (service=%s, exporter=%s)",
        service_name,
        exporter_type,
    )
    return provider


def get_tracer(name: str = "gaokao_tutor") -> trace.Tracer:
    """Return a Tracer instance. Safe to call even if tracing is not initialized."""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the TracerProvider."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracing shut down")
        _tracer_provider = None
