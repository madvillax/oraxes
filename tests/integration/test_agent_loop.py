from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from oraxes.agent.researcher import ResearchAgent
from oraxes.agent.state import AgentState
from oraxes.research.models import (
    AgentEventKind,
    Citation,
    PageSnapshot,
    ResearchFinding,
    ResearchResult,
    ToolCall,
)
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
async def test_loop_collects_verbatim_evidence_and_returns_citations() -> None:
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
    assert result.limitations == []


class RepeatingSearchProvider:
    def __init__(self) -> None:
        self.decisions = 0

    async def decide(self, _: str, __: AgentState, ___: list[ToolDefinition]) -> ScriptedTurn:
        self.decisions += 1
        return ScriptedTurn(
            [
                ToolCall(
                    id=str(self.decisions),
                    name="search",
                    arguments={"query": "program"},
                )
            ]
        )

    async def synthesize(self, question: str, _: AgentState) -> ResearchResult:
        return ResearchResult(question=question, answer="No supported finding was collected.")


@pytest.mark.integration
async def test_loop_synthesizes_at_planning_step_limit() -> None:
    async def html(_: str) -> str:
        return "<html><body>No results</body></html>"

    provider = RepeatingSearchProvider()
    agent = ResearchAgent(
        provider,
        max_steps=2,
        browser_factory=FakeBrowser,
        search=SearchTool(client=html),
    )

    result = await agent.run("What happened?")

    assert provider.decisions == 2
    assert result.limitations == ["Stopped after the configured 2 tool-planning steps."]


class SequenceProvider:
    def __init__(self, batches: list[list[ToolCall]], result: ResearchResult | None = None) -> None:
        self.batches = iter(batches)
        self.decisions: list[AgentState] = []
        self.syntheses: list[AgentState] = []
        self.result = result

    async def decide(self, _: str, state: AgentState, __: list[ToolDefinition]) -> ScriptedTurn:
        self.decisions.append(state.model_copy(deep=True))
        return ScriptedTurn(next(self.batches))

    async def synthesize(self, question: str, state: AgentState) -> ResearchResult:
        self.syntheses.append(state.model_copy(deep=True))
        return self.result or ResearchResult(question=question, answer="Research finished.")


def evidence_batch() -> list[ToolCall]:
    return [
        ToolCall(id="open", name="browser_open", arguments={"url": "https://example.test/source"}),
        ToolCall(
            id="evidence",
            name="collect_evidence",
            arguments={
                "claim": "The program started in 2024.",
                "quote": "The primary source says the program started in 2024.",
                "relevance": "Direct statement from the source.",
            },
        ),
        ToolCall(id="done", name="complete_research", arguments={"reason": "Evidence collected."}),
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    "calls",
    [[], [ToolCall(id="done", name="complete_research", arguments={"reason": "Done."})]],
)
async def test_loop_stops_early_and_synthesizes_once(calls: list[ToolCall]) -> None:
    provider = SequenceProvider([calls])
    result = await ResearchAgent(provider, max_steps=3, browser_factory=FakeBrowser).run("Why?")

    assert len(provider.decisions) == len(provider.syntheses) == 1
    assert provider.syntheses[0].steps == 1
    assert [item.tool_call_id for item in provider.syntheses[0].tool_results] == [
        call.id for call in calls
    ]
    assert result.limitations == []


@pytest.mark.integration
async def test_loop_executes_ordered_batch_on_final_turn_and_filters_fabricated_citations() -> None:
    model_result = ResearchResult(
        question="When?",
        answer="The program started in 2024.",
        findings=[
            ResearchFinding(claim="Supported", evidence_ids=["S1", "S99"], confidence=0.9),
            ResearchFinding(claim="Unsupported", evidence_ids=["S99"], confidence=0.9),
        ],
        citations=[
            Citation(id="S99", url="https://invented.test/", title="Invented", quote="Made up.")
        ],
    )
    provider = SequenceProvider([evidence_batch()], model_result)
    result = await ResearchAgent(provider, max_steps=1, browser_factory=FakeBrowser).run("When?")

    state = provider.syntheses[0]
    assert [item.tool_call_id for item in state.tool_results] == ["open", "evidence", "done"]
    assert all(not item.is_error for item in state.tool_results)
    assert [event.kind for event in state.events] == [AgentEventKind.TOOL_CALL] * 3 + [
        AgentEventKind.TOOL_RESULT
    ] * 3
    assert [citation.id for citation in result.citations] == ["S1"]
    assert result.findings == [
        ResearchFinding(claim="Supported", evidence_ids=["S1"], confidence=0.9)
    ]
    assert result.limitations == []
    assert len(provider.decisions) == len(provider.syntheses) == 1
    assert model_result.findings[0].evidence_ids == ["S1", "S99"]
    assert model_result.citations[0].id == "S99"


@pytest.mark.integration
async def test_loop_passes_tool_errors_to_next_decision() -> None:
    provider = SequenceProvider(
        [
            [
                ToolCall(
                    id="invalid",
                    name="collect_evidence",
                    arguments={"claim": "Claim", "quote": "Missing", "relevance": "Relevant"},
                )
            ],
            [],
        ]
    )
    result = await ResearchAgent(provider, max_steps=3, browser_factory=FakeBrowser).run("Why?")

    assert len(provider.decisions) == 2
    assert provider.decisions[1].tool_results[0].is_error
    assert provider.decisions[1].events[-1].kind == AgentEventKind.TOOL_RESULT
    assert len(provider.syntheses) == 1
    assert result.citations == []


@pytest.mark.integration
async def test_agent_uses_fresh_state_and_browser_for_each_run() -> None:
    browsers: list[FakeBrowser] = []

    def browser_factory() -> FakeBrowser:
        browser = FakeBrowser()
        browsers.append(browser)
        return browser

    provider = SequenceProvider([evidence_batch(), evidence_batch()])
    agent = ResearchAgent(provider, max_steps=3, browser_factory=browser_factory)
    first = await agent.run("First question")
    second = await agent.run("Second question")

    assert len(browsers) == 2
    assert browsers[0] is not browsers[1]
    assert [state.question for state in provider.decisions] == ["First question", "Second question"]
    assert all(state.steps == 1 and not state.tool_results for state in provider.decisions)
    assert [citation.id for citation in first.citations] == ["S1"]
    assert [citation.id for citation in second.citations] == ["S1"]


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["decide", "synthesize"])
async def test_agent_closes_browser_when_provider_fails(phase: str) -> None:
    class TrackingBrowser(FakeBrowser):
        closed = False

        async def __aexit__(self, *_: object) -> None:
            self.closed = True

    class FailingProvider(SequenceProvider):
        async def decide(
            self, question: str, state: AgentState, tools: list[ToolDefinition]
        ) -> ScriptedTurn:
            if phase == "decide":
                raise RuntimeError("Provider failed")
            return await super().decide(question, state, tools)

        async def synthesize(self, question: str, state: AgentState) -> ResearchResult:
            raise RuntimeError("Provider failed")

    browser = TrackingBrowser()
    agent = ResearchAgent(FailingProvider([[]]), browser_factory=lambda: browser)

    with pytest.raises(RuntimeError, match="Provider failed"):
        await agent.run("Why?")

    assert browser.closed
