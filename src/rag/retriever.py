"""Retrieval interface over the ChromaDB vector store."""

from __future__ import annotations

from typing import Optional

from src.rag.indexer import load_index

RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

_vectorstore = None


def _get_vectorstore():
    """Lazy-load singleton — avoids re-reading ChromaDB from disk on every query."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_index()
    return _vectorstore


def retrieve(
    query: str,
    subject: Optional[str] = None,
    year: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Run semantic similarity search and return results with a hit/miss flag.

    ``similarity_search_with_relevance_scores`` returns *relevance* scores
    where **higher is better** (typically 0-1).  We compare the best score
    against ``RELEVANCE_THRESHOLD`` directly.

    Returns
    -------
    dict with keys:
        docs  : list[dict]  — [{content, source, score}, ...]
        is_hit: bool         — True if best relevance score >= RELEVANCE_THRESHOLD
    """
    vectorstore = _get_vectorstore()

    where_filter: dict | None = None
    conditions: list[dict] = []
    if subject:
        conditions.append({"subject": {"$eq": subject}})
    if year:
        conditions.append({"year": {"$eq": year}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=top_k,
        filter=where_filter,
    )

    docs = []
    for doc, score in results:
        docs.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "score": round(score, 4),
                "metadata": doc.metadata,
            }
        )

    is_hit = bool(docs) and docs[0]["score"] >= RELEVANCE_THRESHOLD

    return {"docs": docs, "is_hit": is_hit}
