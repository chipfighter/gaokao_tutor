"""RAG retrieval wrapped as a LangChain tool for use in LangGraph nodes."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from src.rag.retriever import retrieve


@tool
def rag_retrieve(query: str, subject: Optional[str] = None) -> dict:
    """Search the local Gaokao knowledge base (past exams, syllabi, notes).

    Returns a dict with:
      - docs: list of matching document chunks with content, source, score
      - is_hit: whether the retrieval found relevant results
    """
    return retrieve(query=query, subject=subject)
