from __future__ import annotations

from oraxes.research.models import PageElement, PageSnapshot


class FakeBrowser:
    """Deterministic browser test double with the BrowserSession public surface."""

    def __init__(self) -> None:
        self.url = "https://example.test/"
        self.title = "Example Source"
        self.text = "The primary source says the program started in 2024."
        self.typed = ""

    async def __aenter__(self) -> FakeBrowser:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def navigate(self, url: str) -> PageSnapshot:
        self.url = url
        return await self.read_page()

    async def click(self, _: str) -> PageSnapshot:
        return await self.read_page()

    async def type(self, _: str, text: str) -> None:
        self.typed = text

    async def read_page(self, max_chars: int = 8_000) -> PageSnapshot:
        return PageSnapshot(url=self.url, title=self.title, text=self.text[:max_chars])

    async def extract_elements(self, _: str, limit: int = 30) -> list[PageElement]:
        return [PageElement(text="Example", tag="a", href=self.url)][:limit]

    async def scroll(self, _: int = 700) -> PageSnapshot:
        return await self.read_page()

    async def back(self) -> PageSnapshot:
        return await self.read_page()
