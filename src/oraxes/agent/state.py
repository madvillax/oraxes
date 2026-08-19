"""In-memory V1 state. It intentionally has no persistence or database dependency."""

from __future__ import annotations

from pydantic import BaseModel, Field

from oraxes.research.models import AgentEvent, Citation, Evidence, ToolResult


class AgentState(BaseModel):
    question: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)
    completion_requested: bool = False

    def add_evidence(self, evidence: Evidence) -> None:
        if evidence.citation.id not in {item.citation.id for item in self.evidence}:
            self.evidence.append(evidence)

    @property
    def citations(self) -> list[Citation]:
        return [item.citation for item in self.evidence]
