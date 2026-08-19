"""The custom, bounded async research-agent loop."""

from __future__ import annotations

from collections.abc import Callable

from oraxes.agent.state import AgentState
from oraxes.browser.session import BrowserSession
from oraxes.llm.provider import LLMProvider
from oraxes.research.models import AgentEvent, AgentEventKind, ResearchResult
from oraxes.tools.fetch import PageFetcher
from oraxes.tools.registry import ToolRegistry
from oraxes.tools.search import SearchTool


class ResearchAgent:
    """Coordinates one LLM, one browser, and local tools without an agent framework."""

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
        state = AgentState(question=question)
        async with self._browser_factory() as browser:
            registry = ToolRegistry(browser, self._search, self._fetcher)
            while state.steps < self._max_steps and not state.completion_requested:
                state.steps += 1
                turn = await self._provider.decide(question, state, registry.definitions)
                if not turn.tool_calls:
                    break
                for call in turn.tool_calls:
                    state.events.append(AgentEvent(kind=AgentEventKind.TOOL_CALL, detail=call.name))
                    await registry.execute(call, state)
        result = await self._provider.synthesize(question, state)
        result.citations = state.citations
        valid_ids = {citation.id for citation in result.citations}
        for finding in result.findings:
            finding.evidence_ids = [
                identifier for identifier in finding.evidence_ids if identifier in valid_ids
            ]
        result.findings = [finding for finding in result.findings if finding.evidence_ids]
        if state.steps >= self._max_steps and not state.completion_requested:
            result.limitations.append(
                f"Stopped after the configured {self._max_steps} tool-planning steps."
            )
        return result
