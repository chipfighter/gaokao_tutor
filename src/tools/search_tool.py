"""DuckDuckGo Web Search tool for online fallback retrieval.

Provides a lazy-singleton search tool and a convenience ``search()``
function that normalises DuckDuckGo output to a unified schema
(``content``, ``title``, ``url``) consumed by graph nodes.
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults

_search_tool: DuckDuckGoSearchResults | None = None


def get_search_tool() -> DuckDuckGoSearchResults:
    """Lazy singleton — avoids repeated instantiation across graph nodes."""
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchResults(
            max_results=3,
            output_format="list",
        )
    return _search_tool


def search(query: str) -> list[dict]:
    """Execute a web search and return normalised results.

    DuckDuckGo returns keys ``snippet``, ``title``, ``link``.
    This function remaps them to ``content``, ``title``, ``url``
    so that downstream consumers stay provider-agnostic.

    Args:
        query: The search query string.

    Returns:
        A list of dicts with keys ``content``, ``title``, ``url``.
        Returns an empty list on any failure.
    """
    tool = get_search_tool()
    try:
        results = tool.invoke(query)
    except Exception:
        return []

    if isinstance(results, str):
        return [{"content": results, "title": "", "url": ""}]

    return [
        {
            "content": r.get("snippet", r.get("content", "")),
            "title": r.get("title", ""),
            "url": r.get("link", r.get("url", "")),
        }
        for r in results
    ]
