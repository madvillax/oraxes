"""A small async Playwright abstraction exposed to the agent as browser tools."""

from __future__ import annotations

from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from oraxes.research.models import PageElement, PageSnapshot


class BrowserSession:
    """Owns one isolated Chromium context and a single agent-controlled tab."""

    def __init__(self, *, headless: bool = True, timeout_ms: int = 20_000) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> BrowserSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._page is not None:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self._timeout_ms)

    async def close(self) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = self._browser = self._context = self._page = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserSession.start() must be called before browser actions")
        return self._page

    async def navigate(self, url: str) -> PageSnapshot:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Only absolute http(s) URLs are allowed")
        await self.start()
        await self.page.goto(url, wait_until="domcontentloaded")
        return await self.read_page()

    async def click(self, selector: str) -> PageSnapshot:
        await self.page.locator(selector).first.click()
        await self.page.wait_for_load_state("domcontentloaded")
        return await self.read_page()

    async def type(self, selector: str, text: str) -> None:
        await self.page.locator(selector).first.fill(text)

    async def read_page(self, max_chars: int = 8_000) -> PageSnapshot:
        text = await self.page.locator("body").inner_text()
        return PageSnapshot(url=self.page.url, title=await self.page.title(), text=text[:max_chars])

    async def extract_elements(self, selector: str, limit: int = 30) -> list[PageElement]:
        elements = self.page.locator(selector)
        count = min(await elements.count(), limit)
        extracted: list[PageElement] = []
        for index in range(count):
            element = elements.nth(index)
            extracted.append(
                PageElement(
                    text=(await element.inner_text())[:2_000],
                    tag=await element.evaluate("node => node.tagName.toLowerCase()"),
                    href=await element.get_attribute("href"),
                )
            )
        return extracted

    async def scroll(self, pixels: int = 700) -> PageSnapshot:
        await self.page.evaluate("amount => window.scrollBy(0, amount)", pixels)
        return await self.read_page()

    async def back(self) -> PageSnapshot:
        await self.page.go_back(wait_until="domcontentloaded")
        return await self.read_page()
