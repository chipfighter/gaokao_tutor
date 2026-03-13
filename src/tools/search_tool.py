"""Tavily Web Search tool for online fallback retrieval."""

from __future__ import annotations

import os

from langchain_community.tools.tavily_search import TavilySearchResults

_search_tool = None


def get_search_tool() -> TavilySearchResults:
    """Lazy singleton — ensures env vars are loaded before instantiation."""
    global _search_tool
    if _search_tool is None:
        _search_tool = TavilySearchResults(
            max_results=3,
            api_key=os.getenv("TAVILY_API_KEY"),
        )
    return _search_tool
