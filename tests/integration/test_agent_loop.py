from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from oraxes.agent.researcher import ResearchAgent
from oraxes.agent.state import AgentState
from oraxes.research.models import PageSnapshot, ResearchFinding, ResearchResult, ToolCall
from oraxes.tools.fetch import PageFetcher
from oraxes.tools.registry import ToolDefinition
from oraxes.tools.search import SearchTool


class FakeBrowser:
    def __init__(self) -> None:
        self.url = "https://example.test/"
        self.title = "Example Source"
        self.text = "The primary source says the program started in 2024."

    async def __aenter__(self) -> FakeBrowser:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def navigate(self, url: str) -> PageSnapshot:
        self.url = url
        return await self.read_page()

    async def read_page(self, max_chars: int = 8_000) -> PageSnapshot:
        return PageSnapshot(url=self.url, title=self.title, text=self.text[:max_chars])


@dataclass
class ScriptedTurn:
    tool_calls: list[ToolCall]
    text: str = ""


class ScriptedProvider:
    async def decide(self, _: str, state: AgentState, __: list[ToolDefinition]) -> ScriptedTurn:
        if not state.tool_results:
            return ScriptedTurn([ToolCall(id="1", name="search", arguments={"query": "program"})])
        if len(state.tool_results) == 1:
            return ScriptedTurn(
                [
                    ToolCall(
                        id="2",
                        name="fetch_page",
                        arguments={"url": "https://example.test/source"},
                    )
                ]
            )
        if len(state.tool_results) == 2:
            return ScriptedTurn(
                [
                    ToolCall(
                        id="3",
                        name="collect_evidence",
                        arguments={
                            "claim": "The program started in 2024.",
                            "quote": "The primary source says the program started in 2024.",
                            "relevance": "Direct statement from the source.",
                        },
                    )
                ]
            )
        return ScriptedTurn(
            [
                ToolCall(
                    id="4", name="complete_research", arguments={"reason": "Evidence collected."}
                )
            ]
        )

    async def synthesize(self, question: str, _: AgentState) -> ResearchResult:
        return ResearchResult(
            question=question,
            answer="The program started in 2024.",
            findings=[
                ResearchFinding(claim="It started in 2024.", evidence_ids=["S1"], confidence=0.9)
            ],
        )


@pytest.mark.integration
async def test_agent_collects_verbatim_evidence_and_returns_citations() -> None:
    async def html(_: str) -> str:
        return "<article class='result'><a class='result__a' href='https://example.test/source'>Source</a></article>"

    def fetch_response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=(
                "<title>Example Source</title><p>The primary source says the program "
                "started in 2024.</p>"
            ),
        )

    agent = ResearchAgent(
        ScriptedProvider(),
        max_steps=8,
        browser_factory=FakeBrowser,
        search=SearchTool(client=html),
        fetcher=PageFetcher(transport=httpx.MockTransport(fetch_response)),
    )
    result = await agent.run("When did the program start?")

    assert result.citations[0].id == "S1"
    assert result.citations[0].quote == "The primary source says the program started in 2024."
    assert result.findings[0].evidence_ids == ["S1"]
