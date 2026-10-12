"""Application-owned async loop for bounded, evidence-backed research."""

from __future__ import annotations

from oraxes.agent.state import AgentState
from oraxes.llm.provider import LLMProvider
from oraxes.research.models import AgentEvent, AgentEventKind, ResearchResult
from oraxes.tools.registry import ToolRegistry


class ResearchLoop:
    """Plan, execute tools in order, and synthesize once research stops."""

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

    async def run(self, question: str) -> ResearchResult:
        state = AgentState(question=question)

        for _ in range(self._max_steps):
            state.steps += 1
            turn = await self._provider.decide(
                state.question,
                state,
                self._registry.definitions,
            )
            calls = list(turn.tool_calls)
            if not calls:
                break

            state.events.extend(
                AgentEvent(kind=AgentEventKind.TOOL_CALL, detail=call.name) for call in calls
            )
            # Tools share a page snapshot, so finish each batch in the requested order.
            for call in calls:
                await self._registry.execute(call, state)

            if state.completion_requested:
                break

        return await self._synthesize(state)

    async def _synthesize(self, state: AgentState) -> ResearchResult:
        result = await self._provider.synthesize(state.question, state)
        result = result.model_copy(deep=True)
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
