"""Public research-agent facade backed by a LangGraph workflow."""

from __future__ import annotations

from collections.abc import Callable

from oraxes.agent.graph import ResearchGraph
from oraxes.browser.session import BrowserSession
from oraxes.llm.provider import LLMProvider
from oraxes.research.models import ResearchResult
from oraxes.tools.fetch import PageFetcher
from oraxes.tools.registry import ToolRegistry
from oraxes.tools.search import SearchTool


class ResearchAgent:
    """Create per-run dependencies and execute the LangGraph research workflow."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_steps: int = 12,
        browser_factory: Callable[[], BrowserSession] = BrowserSession,
        search: SearchTool | None = None,
        fetcher: PageFetcher | None = None,
    ) -> None:
        self._provider = provider
        self._max_steps = max_steps
        self._browser_factory = browser_factory
        self._search = search
        self._fetcher = fetcher

    async def run(self, question: str) -> ResearchResult:
        async with self._browser_factory() as browser:
            registry = ToolRegistry(browser, self._search, self._fetcher)
            graph = ResearchGraph(
                self._provider,
                registry,
                max_steps=self._max_steps,
            )
            return await graph.run(question)
