"""Hybrid retrieval: vector search + BM25 keyword search + BGE reranker."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import get_setting
from src.rag.indexer import load_index
from src.rag.reranker import rerank

logger = logging.getLogger(__name__)

# Legacy constants kept for backward compatibility with existing tests
RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Singletons (lazy-loaded)
# ---------------------------------------------------------------------------

_vectorstore = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict[str, Any]] = []  # parallel list of doc dicts


def _get_vectorstore():
    """Lazy-load singleton — avoids re-reading ChromaDB from disk on every query."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_index()
    return _vectorstore


def _build_bm25_index() -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """Build a BM25 index from all documents stored in ChromaDB.

    Returns (bm25_index, corpus) where corpus is a parallel list of doc dicts
    with keys: content, source, metadata.
    """
    try:
        vs = _get_vectorstore()
        collection = vs._collection
        data = collection.get(include=["documents", "metadatas"])

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        if not documents:
            logger.warning("ChromaDB collection is empty; BM25 index will be empty")
            return None, []

        corpus: list[dict[str, Any]] = []
        tokenized: list[list[str]] = []

        for doc_text, meta in zip(documents, metadatas):
            if not doc_text:
                continue
            corpus.append({
                "content": doc_text,
                "source": (meta or {}).get("source_file", "unknown"),
                "metadata": meta or {},
            })
            tokenized.append(jieba.lcut(doc_text))

        if not tokenized:
            return None, []

        return BM25Okapi(tokenized), corpus

    except Exception:
        logger.warning("Failed to build BM25 index; keyword search disabled", exc_info=True)
        return None, []


def _get_bm25() -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    """Lazy-load BM25 singleton."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None and not _bm25_corpus:
        _bm25_index, _bm25_corpus = _build_bm25_index()
    return _bm25_index, _bm25_corpus


def _bm25_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Run BM25 keyword search over the cached corpus."""
    bm25, corpus = _get_bm25()
    if bm25 is None or not corpus:
        return []

    tokens = jieba.lcut(query)
    scores = bm25.get_scores(tokens)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results: list[dict[str, Any]] = []
    for idx, score in scored[:top_k]:
        if score <= 0:
            break
        doc = corpus[idx]
        results.append({
            "content": doc["content"],
            "source": doc["source"],
            "score": round(float(score), 4),
            "metadata": doc["metadata"],
        })
    return results


def _content_hash(text: str) -> str:
    """MD5 hash for dedup."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _merge_and_dedup(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge vector + BM25 results, deduplicate by content hash."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # Vector results first (they have calibrated relevance scores)
    for doc in vector_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    for doc in bm25_results:
        h = _content_hash(doc["content"])
        if h not in seen:
            seen.add(h)
            merged.append(doc)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    subject: Optional[str] = None,
    year: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Hybrid retrieval: vector search + BM25 + reranker.

    Returns
    -------
    dict with keys:
        docs  : list[dict]  — [{content, source, score, metadata}, ...]
        is_hit: bool         — True if best score >= relevance threshold
    """
    vector_top_k = get_setting("rag.vector_top_k", 10)
    bm25_top_k = get_setting("rag.bm25_top_k", 10)
    reranker_top_n = get_setting("rag.reranker_top_n", top_k)
    threshold = get_setting("rag.relevance_threshold", RELEVANCE_THRESHOLD)

    # --- 1. Vector search (existing) ---
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
        k=vector_top_k,
        filter=where_filter,
    )

    vector_docs: list[dict[str, Any]] = []
    for doc, score in results:
        vector_docs.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "unknown"),
            "score": round(score, 4),
            "metadata": doc.metadata,
        })

    # --- 2. BM25 keyword search ---
    bm25_docs = _bm25_search(query, top_k=bm25_top_k)

    # --- 3. Merge + deduplicate ---
    merged = _merge_and_dedup(vector_docs, bm25_docs)

    # --- 4. Rerank ---
    if merged:
        ranked = rerank(query, merged, top_n=reranker_top_n)
    else:
        ranked = []

    # --- 5. Determine hit ---
    is_hit = False
    if ranked:
        # Use rerank_score if available, else original score
        best_score = ranked[0].get("rerank_score", ranked[0].get("score", 0))
        is_hit = best_score >= threshold

    return {"docs": ranked, "is_hit": is_hit}
