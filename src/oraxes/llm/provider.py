"""Provider interface: the agent depends on this, not on Gemini directly."""

from __future__ import annotations

from typing import Protocol

from oraxes.agent.state import AgentState
from oraxes.research.models import ResearchResult, ToolCall
from oraxes.tools.registry import ToolDefinition


class ProviderTurn(Protocol):
    @property
    def tool_calls(self) -> list[ToolCall]: ...

    @property
    def text(self) -> str: ...


class LLMProvider(Protocol):
    """The provider-neutral contract used by the LangGraph research workflow."""

    async def decide(
        self, question: str, state: AgentState, tools: list[ToolDefinition]
    ) -> ProviderTurn: ...

    async def synthesize(self, question: str, state: AgentState) -> ResearchResult: ...
