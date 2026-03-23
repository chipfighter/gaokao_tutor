"""SiliconFlow BGE Reranker API wrapper.

Calls the SiliconFlow reranking endpoint to re-score candidate documents
against a query.  On any API failure the original documents are returned
in their existing order (graceful degradation).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.config import get_setting

logger = logging.getLogger(__name__)

_DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
_TIMEOUT = 15  # seconds


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank *documents* against *query* via SiliconFlow BGE Reranker.

    Parameters
    ----------
    query : str
        The user's search query.
    documents : list[dict]
        Candidate documents.  Each dict **must** have a ``"content"`` key.
    top_n : int, optional
        Number of results to return.  Defaults to ``rag.reranker_top_n``
        from settings (fallback: 5).

    Returns
    -------
    list[dict]
        The *top_n* documents sorted by reranker relevance, each with an
        added ``"rerank_score"`` key.  On failure, returns *documents*
        truncated to *top_n* in original order.
    """
    if not documents:
        return []

    if top_n is None:
        top_n = get_setting("rag.reranker_top_n", 5)

    api_key = os.getenv("SILICONFLOW_API_KEY")
    model = os.getenv(
        "RERANKER_MODEL",
        get_setting("rag.reranker_model", _DEFAULT_RERANKER_MODEL),
    )

    doc_texts = [d["content"] for d in documents]

    try:
        resp = httpx.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "query": query,
                "documents": doc_texts,
                "top_n": top_n,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Reranker API call failed; returning original order", exc_info=True)
        return documents[:top_n]

    results: list[dict[str, Any]] = data.get("results", [])
    ranked: list[dict[str, Any]] = []
    for item in results:
        idx = item["index"]
        if 0 <= idx < len(documents):
            doc = {**documents[idx], "rerank_score": item["relevance_score"]}
            ranked.append(doc)

    return ranked[:top_n]
