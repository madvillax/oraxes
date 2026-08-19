"""A bounded web search tool based on DuckDuckGo's HTML endpoint."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from bs4 import BeautifulSoup

from oraxes.research.models import SourceSearchResult

SearchClient = Callable[[str], Awaitable[str]]


class SearchTool:
    name = "search"
    description = "Search the public web for sources relevant to the research question."

    def __init__(self, client: SearchClient | None = None) -> None:
        self._client = client or self._request_html

    async def _request_html(self, query: str) -> str:
        url = f"https://html.duckduckgo.com/html/?{urlencode({'q': query})}"
        async with httpx.AsyncClient(
            headers={"User-Agent": "OraxesResearch/0.1 (+https://github.com/)"},
            follow_redirects=True,
        ) as client:
            response = await client.get(url, timeout=20)
            response.raise_for_status()
            return response.text

    async def run(self, query: str, max_results: int = 5) -> list[SourceSearchResult]:
        soup = BeautifulSoup(await self._client(query), "html.parser")
        results: list[SourceSearchResult] = []
        for result in soup.select(".result"):
            link = result.select_one(".result__a")
            if link is None or not link.get("href"):
                continue
            snippet_node = result.select_one(".result__snippet")
            try:
                results.append(
                    SourceSearchResult(
                        title=link.get_text(" ", strip=True),
                        url=self._normalize_result_url(str(link["href"])),
                        snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    )
                )
            except ValueError:
                continue
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _normalize_result_url(raw_url: str) -> str:
        absolute_url = f"https:{raw_url}" if raw_url.startswith("//") else raw_url
        parsed = urlparse(absolute_url)
        if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
            destinations = parse_qs(parsed.query).get("uddg")
            if destinations:
                return destinations[0]
        return absolute_url
