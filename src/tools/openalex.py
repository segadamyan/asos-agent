from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


OPENALEX_BASE_URL = "https://api.openalex.org"


class OpenAlexError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    """
    Minimal tool spec that works with most custom agent frameworks:
    - name: tool name exposed to the LLM
    - description: what it does
    - parameters: JSON schema-like dict
    - handler: callable that executes the tool
    - category: used by your ToolsFactory grouping (optional)
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    category: str = "academic"


def _http_get_json(
    url: str,
    timeout_s: float = 15.0,
    retries: int = 2,
    backoff_s: float = 0.6,
    user_agent: str = "asos-agent-openalex-tool/1.0",
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    headers = {
        "Accept": "application/json",
        "User-Agent": user_agent,
    }

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_s * (2 ** attempt))
            else:
                break

    raise OpenAlexError(f"OpenAlex request failed after retries. url={url} err={last_err}")


def _build_url(path: str, params: Dict[str, Any]) -> str:
    clean_params: Dict[str, str] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            clean_params[k] = "true" if v else "false"
        else:
            clean_params[k] = str(v)

    qs = urllib.parse.urlencode(clean_params, doseq=True)
    return f"{OPENALEX_BASE_URL}{path}?{qs}" if qs else f"{OPENALEX_BASE_URL}{path}"


def _compact_work(work: Dict[str, Any]) -> Dict[str, Any]:
    authorships = work.get("authorships") or []
    authors = []
    for a in authorships[:10]:
        author = (a.get("author") or {})
        authors.append({
            "id": author.get("id"),
            "display_name": author.get("display_name"),
        })

    primary_location = work.get("primary_location") or {}
    source = (primary_location.get("source") or {})

    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "publication_year": work.get("publication_year"),
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "open_access": (work.get("open_access") or {}).get("is_oa"),
        "source": {
            "id": source.get("id"),
            "display_name": source.get("display_name"),
        },
        "authors": authors,
        # helpful for ranking without huge payloads:
        "concepts": [
            {"id": c.get("id"), "display_name": c.get("display_name"), "score": c.get("score")}
            for c in (work.get("concepts") or [])[:8]
        ],
        "url": work.get("id"),  # OpenAlex ID is also URL
    }


def _compact_author(author: Dict[str, Any]) -> Dict[str, Any]:
    insts = author.get("last_known_institutions") or []
    return {
        "id": author.get("id"),
        "orcid": author.get("orcid"),
        "display_name": author.get("display_name"),
        "works_count": author.get("works_count"),
        "cited_by_count": author.get("cited_by_count"),
        "last_known_institutions": [
            {"id": i.get("id"), "display_name": i.get("display_name"), "country_code": i.get("country_code")}
            for i in insts[:5]
        ],
        "url": author.get("id"),
    }


# ------------------------
# Tool handlers (callables)
# ------------------------

def openalex_search_works(
    query: str,
    per_page: int = 5,
    sort: str = "cited_by_count:desc",
    filter: Optional[str] = None,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Search scholarly works (papers).
    - query: full-text search
    - sort: e.g. 'cited_by_count:desc' or 'publication_date:desc'
    - filter: OpenAlex filter string, e.g. 'publication_year:2020-2025,open_access.is_oa:true'
    """
    per_page = max(1, min(int(per_page), 25))
    params = {
        "search": query,
        "per-page": per_page,
        "sort": sort,
        "filter": filter,
    }
    url = _build_url("/works", params)
    data = _http_get_json(url, timeout_s=timeout_s, retries=retries)

    results = data.get("results") or []
    return {
        "meta": data.get("meta", {}),
        "results": [_compact_work(w) for w in results],
        "request": {"url": url},
    }


def openalex_get_work(
    work_id_or_doi: str,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Fetch a single work by:
    - OpenAlex ID: https://openalex.org/Wxxxx
    - DOI (either full 'https://doi.org/...' or bare '10.xxxx/yyy')
    """
    x = work_id_or_doi.strip()

    if x.lower().startswith("https://openalex.org/"):
        # convert to /works/Wxxxx
        work_id = x.rsplit("/", 1)[-1]
        path = f"/works/{work_id}"
        url = _build_url(path, {})
    elif x.lower().startswith("doi:"):
        doi = x[4:]
        url = _build_url(f"/works/doi:{doi}", {})
    elif x.lower().startswith("10."):
        url = _build_url(f"/works/doi:{x}", {})
    else:
        # allow passing Wxxxx
        if x.startswith("W"):
            url = _build_url(f"/works/{x}", {})
        else:
            raise OpenAlexError(
                "openalex_get_work expects an OpenAlex work id (Wxxxx), an OpenAlex URL, or a DOI."
            )

    data = _http_get_json(url, timeout_s=timeout_s, retries=retries)
    return {
        "result": _compact_work(data),
        "request": {"url": url},
    }


def openalex_search_authors(
    query: str,
    per_page: int = 5,
    sort: str = "cited_by_count:desc",
    filter: Optional[str] = None,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> Dict[str, Any]:
    per_page = max(1, min(int(per_page), 25))
    params = {
        "search": query,
        "per-page": per_page,
        "sort": sort,
        "filter": filter,
    }
    url = _build_url("/authors", params)
    data = _http_get_json(url, timeout_s=timeout_s, retries=retries)

    results = data.get("results") or []
    return {
        "meta": data.get("meta", {}),
        "results": [_compact_author(a) for a in results],
        "request": {"url": url},
    }


def openalex_get_author(
    author_id_or_orcid: str,
    timeout_s: float = 15.0,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Fetch author by:
    - OpenAlex Author ID: Axxxx
    - OpenAlex URL: https://openalex.org/Axxxx
    - ORCID: '0000-0002-....' or 'orcid:0000-....'
    """
    x = author_id_or_orcid.strip()

    if x.lower().startswith("https://openalex.org/"):
        author_id = x.rsplit("/", 1)[-1]
        url = _build_url(f"/authors/{author_id}", {})
    elif x.lower().startswith("orcid:"):
        url = _build_url(f"/authors/orcid:{x[6:]}", {})
    elif len(x) >= 15 and x.count("-") >= 3:
        # looks like ORCID
        url = _build_url(f"/authors/orcid:{x}", {})
    else:
        if x.startswith("A"):
            url = _build_url(f"/authors/{x}", {})
        else:
            raise OpenAlexError(
                "openalex_get_author expects an OpenAlex author id (Axxxx), an OpenAlex URL, or an ORCID."
            )

    data = _http_get_json(url, timeout_s=timeout_s, retries=retries)
    return {
        "result": _compact_author(data),
        "request": {"url": url},
    }


# ------------------------
# Export tool specs
# ------------------------

def get_openalex_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="openalex_search_works",
            description=(
                "Search scholarly papers (works) using OpenAlex. "
                "Use this to find papers by keywords, then rank by citations or recency."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Full-text search query"},
                    "per_page": {"type": "integer", "description": "Results to return (1-25)", "default": 5},
                    "sort": {
                        "type": "string",
                        "description": "Sort string, e.g. 'cited_by_count:desc' or 'publication_date:desc'",
                        "default": "cited_by_count:desc",
                    },
                    "filter": {
                        "type": "string",
                        "description": "OpenAlex filter expression, e.g. 'publication_year:2020-2025,open_access.is_oa:true'",
                    },
                    "timeout_s": {"type": "number", "default": 15.0},
                    "retries": {"type": "integer", "default": 2},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=openalex_search_works,
            category="academic",
        ),
        ToolSpec(
            name="openalex_get_work",
            description="Get one paper by OpenAlex ID/URL (Wxxxx) or DOI.",
            parameters={
                "type": "object",
                "properties": {
                    "work_id_or_doi": {"type": "string", "description": "OpenAlex ID/URL or DOI"},
                    "timeout_s": {"type": "number", "default": 15.0},
                    "retries": {"type": "integer", "default": 2},
                },
                "required": ["work_id_or_doi"],
                "additionalProperties": False,
            },
            handler=openalex_get_work,
            category="academic",
        ),
        ToolSpec(
            name="openalex_search_authors",
            description="Search authors in OpenAlex (useful for finding an author's OpenAlex ID).",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Author name query"},
                    "per_page": {"type": "integer", "description": "Results to return (1-25)", "default": 5},
                    "sort": {
                        "type": "string",
                        "description": "Sort string, e.g. 'cited_by_count:desc'",
                        "default": "cited_by_count:desc",
                    },
                    "filter": {"type": "string", "description": "OpenAlex filter expression"},
                    "timeout_s": {"type": "number", "default": 15.0},
                    "retries": {"type": "integer", "default": 2},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=openalex_search_authors,
            category="academic",
        ),
        ToolSpec(
            name="openalex_get_author",
            description="Get one author by OpenAlex ID/URL (Axxxx) or ORCID.",
            parameters={
                "type": "object",
                "properties": {
                    "author_id_or_orcid": {"type": "string", "description": "OpenAlex ID/URL or ORCID"},
                    "timeout_s": {"type": "number", "default": 15.0},
                    "retries": {"type": "integer", "default": 2},
                },
                "required": ["author_id_or_orcid"],
                "additionalProperties": False,
            },
            handler=openalex_get_author,
            category="academic",
        ),
    ]
