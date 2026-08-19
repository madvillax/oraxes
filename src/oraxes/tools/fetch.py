"""Fast static-page retrieval used before launching a rendered browser."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from oraxes.research.models import PageSnapshot


class PageFetcher:
    """Fetches and extracts readable HTML without paying the browser startup cost."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def fetch(self, url: str, max_chars: int = 8_000) -> PageSnapshot:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Only absolute http(s) URLs are allowed")
        async with httpx.AsyncClient(
            transport=self._transport,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OraxesResearch/0.1)"},
            follow_redirects=True,
            timeout=15,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and not content_type.startswith("text/"):
            raise ValueError(
                f"fetch_page supports text and HTML, received {content_type or 'unknown'}"
            )
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else str(response.url)
        text = "\n".join(line for line in soup.get_text("\n", strip=True).splitlines() if line)
        return PageSnapshot(url=str(response.url), title=title, text=text[:max_chars])
