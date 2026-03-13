"""Tavily Web Search tool for online fallback retrieval."""

from __future__ import annotations

import os

from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY"),
)
