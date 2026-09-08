"""LangGraph orchestration for the bounded research workflow."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from oraxes.agent.state import AgentState
from oraxes.llm.provider import LLMProvider
from oraxes.research.models import AgentEvent, AgentEventKind, ResearchResult, ToolCall
from oraxes.tools.registry import ToolRegistry


class ResearchGraphState(BaseModel):
    """State passed between LangGraph nodes for one in-memory research run."""

    agent: AgentState
    pending_calls: list[ToolCall]
    result: ResearchResult | None


class ResearchGraphUpdate(TypedDict, total=False):
    """Partial state update returned by a LangGraph node."""

    agent: AgentState
    pending_calls: list[ToolCall]
    result: ResearchResult | None


class ResearchGraph:
    """Compile and run the plan -> tools -> synthesize research graph."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        max_steps: int,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._provider = provider
        self._registry = registry
        self._max_steps = max_steps
        self._compiled = self._build()

    def _build(self):
        builder = StateGraph(ResearchGraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("tools", self._execute_tools)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"tools": "tools", "synthesize": "synthesize"},
        )
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"plan": "plan", "synthesize": "synthesize"},
        )
        builder.add_edge("synthesize", END)
        return builder.compile()

    async def run(self, question: str) -> ResearchResult:
        initial = ResearchGraphState(
            agent=AgentState(question=question),
            pending_calls=[],
            result=None,
        )
        final = await self._compiled.ainvoke(
            initial,
            config={"recursion_limit": (self._max_steps * 2) + 4},
        )
        result = final["result"]
        if result is None:
            raise RuntimeError("Research graph ended without a synthesized result")
        return result

    async def _plan(self, state: ResearchGraphState) -> ResearchGraphUpdate:
        agent = state.agent.model_copy(deep=True)
        agent.steps += 1
        turn = await self._provider.decide(
            agent.question,
            agent,
            self._registry.definitions,
        )
        pending_calls = list(turn.tool_calls)
        agent.events.extend(
            AgentEvent(kind=AgentEventKind.TOOL_CALL, detail=call.name) for call in pending_calls
        )
        return {"agent": agent, "pending_calls": pending_calls}

    async def _execute_tools(self, state: ResearchGraphState) -> ResearchGraphUpdate:
        agent = state.agent.model_copy(deep=True)
        for call in state.pending_calls:
            await self._registry.execute(call, agent)
        return {"agent": agent, "pending_calls": []}

    def _route_after_plan(self, state: ResearchGraphState) -> Literal["tools", "synthesize"]:
        return "tools" if state.pending_calls else "synthesize"

    def _route_after_tools(self, state: ResearchGraphState) -> Literal["plan", "synthesize"]:
        agent = state.agent
        if agent.completion_requested or agent.steps >= self._max_steps:
            return "synthesize"
        return "plan"

    async def _synthesize(self, state: ResearchGraphState) -> ResearchGraphUpdate:
        agent = state.agent
        result = await self._provider.synthesize(agent.question, agent)
        result = result.model_copy(deep=True)
        result.citations = agent.citations

        valid_ids = {citation.id for citation in result.citations}
        for finding in result.findings:
            finding.evidence_ids = [
                identifier for identifier in finding.evidence_ids if identifier in valid_ids
            ]
        result.findings = [finding for finding in result.findings if finding.evidence_ids]

        if agent.steps >= self._max_steps and not agent.completion_requested:
            result.limitations.append(
                f"Stopped after the configured {self._max_steps} tool-planning steps."
            )
        return {"result": result}
