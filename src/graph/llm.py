"""Central LLM factory and fallback invoke logic.

Provides a resilient invoke_with_fallback() that catches transient API errors
(timeouts, 502s, rate limits) and retries on a fallback model, recording the
failover event on the active OpenTelemetry span.
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recoverable errors that trigger automatic fallback
# ---------------------------------------------------------------------------

_FALLBACK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

try:
    import openai

    _FALLBACK_ERRORS = (
        TimeoutError,
        ConnectionError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.RateLimitError,
    )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

def get_primary_llm(**overrides) -> ChatOpenAI:
    """Build the primary chat model from DEEPSEEK_* env vars."""
    defaults = dict(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_fallback_llm(**overrides) -> ChatOpenAI:
    """Build the fallback chat model from FALLBACK_* env vars.

    Defaults to the primary API config so that transient errors (502, timeout)
    get a second chance on the same endpoint.  Override ``FALLBACK_MODEL``,
    ``FALLBACK_API_KEY``, and ``FALLBACK_BASE_URL`` to point at a local Ollama
    instance or a different cloud provider.
    """
    defaults = dict(
        model=os.getenv("FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        api_key=os.getenv("FALLBACK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "not-configured",
        base_url=os.getenv("FALLBACK_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        temperature=0.7,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


# ---------------------------------------------------------------------------
# Resilient invoke
# ---------------------------------------------------------------------------

def invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """Invoke *primary*; on recoverable error, failover to *fallback*.

    Args:
        primary: Primary ChatModel instance.
        messages: Message list passed to ``invoke()``.
        fallback: Optional fallback ChatModel. ``None`` → error propagates.
        span: Optional OTel span for recording fallback metadata.

    Returns:
        The LLM response from whichever model succeeded.

    Raises:
        The original error when no fallback is configured, or the fallback
        error when both models fail.
    """
    try:
        response = primary.invoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        return fallback.invoke(messages)
