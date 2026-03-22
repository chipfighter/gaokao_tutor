"""ChromaDB index builder with incremental upsert support.

Uses SiliconFlow's OpenAI-compatible embedding API (BAAI/bge-m3) instead
of local HuggingFace models, eliminating heavy local dependencies.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import math

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

COLLECTION_NAME = "gaokao_docs"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def _l2_to_relevance(distance: float) -> float:
    """Convert Chroma L2 distance to a [0, 1] relevance score.

    Chroma's default distance metric is L2 (Euclidean).  For normalized
    embeddings the maximum possible L2 distance is sqrt(2).  We linearly
    map [0, sqrt(2)] → [1, 0] so that higher scores mean higher relevance
    and the values are always within [0, 1].
    """
    return 1.0 - distance / math.sqrt(2)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_persist_dir(persist_directory: Optional[str] = None) -> str:
    """Always resolve to an absolute path anchored at project root."""
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _get_embedding(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """Create an OpenAI-compatible embedding client backed by SiliconFlow.

    Args:
        model_name: Override for the embedding model identifier.
            Falls back to ``EMBEDDING_MODEL`` env var, then
            ``DEFAULT_EMBEDDING_MODEL``.

    Returns:
        A configured ``OpenAIEmbeddings`` instance pointing at SiliconFlow.
    """
    model_name = model_name or os.getenv(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base=os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
        ),
    )


def _content_id(doc: Document) -> str:
    """Deterministic ID from chunk content — true dedup across repeated runs."""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return f"{doc.metadata.get('source_file', 'unknown')}_{digest}"


def build_index(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Create (or update) a ChromaDB collection from *documents*.

    Uses md5 hash of chunk content as the dedup id so repeated runs are safe.

    Args:
        documents: List of LangChain Document objects to index.
        persist_directory: Override for the ChromaDB persistence path.
        embedding_model: Override for the embedding model identifier.

    Returns:
        The populated Chroma vectorstore instance.
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    ids = [_content_id(doc) for doc in documents]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        ids=ids,
        relevance_score_fn=_l2_to_relevance,
    )
    return vectorstore


def load_index(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Args:
        persist_directory: Override for the ChromaDB persistence path.
        embedding_model: Override for the embedding model identifier.

    Returns:
        The loaded Chroma vectorstore instance.
    """
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory,
        relevance_score_fn=_l2_to_relevance,
    )
