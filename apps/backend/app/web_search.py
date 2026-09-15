from __future__ import annotations

import os
from typing import Any

import httpx


TAVILY_URL = "https://api.tavily.com/search"


async def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web through Tavily when configured.

    Web pages are treated as untrusted evidence by the caller; their text is
    never interpreted here as application instructions.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"enabled": False, "results": [], "error": None}

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(TAVILY_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"enabled": True, "results": [], "error": str(exc)}

    results: list[dict[str, Any]] = []
    for index, item in enumerate(data.get("results", []), start=1):
        results.append(
            {
                "id": f"W{index}",
                "source_type": "web",
                "title": item.get("title", "Untitled source"),
                "url": item.get("url", ""),
                "text": item.get("content", ""),
                "published_at": item.get("published_date"),
                "score": item.get("score"),
            }
        )

    return {"enabled": True, "results": results, "error": None}
