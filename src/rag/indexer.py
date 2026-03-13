"""ChromaDB index builder with incremental upsert support."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION_NAME = "gaokao_docs"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_persist_dir(persist_directory: Optional[str] = None) -> str:
    """Always resolve to an absolute path anchored at project root."""
    rel = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "chroma_store/")
    path = Path(rel)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def _get_embedding(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=model_name)


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
    )
    return vectorstore


def load_index(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    persist_directory = _resolve_persist_dir(persist_directory)
    embedding = _get_embedding(embedding_model)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=persist_directory,
    )
