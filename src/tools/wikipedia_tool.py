"""
Wikipedia client (no official API key needed)
Uses MediaWiki API for search + Wikipedia REST API for summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from agents.tools.base import ToolDefinition


@dataclass(frozen=True)
class WikiSummary:
    title: str
    extract: str
    url: str
    description: Optional[str] = None
    page_id: Optional[int] = None
    lang: str = "en"


class WikipediaClient:
    def __init__(
        self,
        *,
        timeout_s: float = 10.0,
        user_agent: str = "NLPProjectWikiTool/1.0 (contact: your-team@example.com)",
    ) -> None:
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def search_title(self, query: str, *, lang: str = "en") -> str:
        """Uses MediaWiki API search to get best matching title."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query.strip(),
            "srlimit": 1,
            "format": "json",
        }

        resp = self._session.get(endpoint, params=params, timeout=self._timeout_s)
        resp.raise_for_status()
        data = resp.json()

        results = (data.get("query", {}) or {}).get("search", []) or []
        if not results:
            raise LookupError(f"No Wikipedia results found for query: {query!r}")

        return results[0]["title"]

    def get_summary(self, title: str, *, lang: str = "en") -> WikiSummary:
        """Uses Wikipedia REST API summary endpoint."""
        if not title or not title.strip():
            raise ValueError("title must be a non-empty string")

        safe_title = title.strip().replace(" ", "_")
        endpoint = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

        resp = self._session.get(endpoint, timeout=self._timeout_s)
        resp.raise_for_status()
        data = resp.json()

        page_type = data.get("type")
        extract = data.get("extract") or ""
        url = (data.get("content_urls", {}) or {}).get("desktop", {}).get("page") or ""

        if page_type == "disambiguation":
            return WikiSummary(
                title=data.get("title") or title,
                extract=(extract + "\n\n(Note: this page is a disambiguation page.)").strip(),
                url=url or f"https://{lang}.wikipedia.org/wiki/{safe_title}",
                description=data.get("description"),
                page_id=data.get("pageid"),
                lang=lang,
            )

        if not extract:
            raise LookupError(f"Summary not available for title: {title!r}")

        return WikiSummary(
            title=data.get("title") or title,
            extract=extract.strip(),
            url=url or f"https://{lang}.wikipedia.org/wiki/{safe_title}",
            description=data.get("description"),
            page_id=data.get("pageid"),
            lang=lang,
        )

    def search_and_summarize(self, query: str, *, lang: str = "en") -> WikiSummary:
        title = self.search_title(query, lang=lang)
        return self.get_summary(title, lang=lang)


# IMPORTANT: create client only AFTER class definition
_client = WikipediaClient()


async def wikipedia_search(query: str, lang: str = "en") -> Dict[str, Any]:
    """Search Wikipedia and return a short summary."""
    try:
        summary = _client.search_and_summarize(query, lang=lang)
        return {
            "title": summary.title,
            "summary": summary.extract,
            "url": summary.url,
            "description": summary.description,
            "lang": summary.lang,
        }
    except Exception as e:
        return {"error": str(e)}


def make_wikipedia_tool() -> ToolDefinition:
    return ToolDefinition(
        name="wikipedia",
        description=(
            "Searches Wikipedia and returns a concise, reliable summary of a topic. "
            "Best for definitions, background knowledge, and factual context."
        ),
        args_description={
            "query": "Topic or concept to search for (e.g. 'Quantum mechanics').",
            "lang": "Wikipedia language code (default: 'en').",
        },
        args_schema={
            "query": {"type": "string"},
            "lang": {"type": "string"},
        },
        tool=wikipedia_search,
    )


def get_wikipedia_tools():
    return [make_wikipedia_tool()]
