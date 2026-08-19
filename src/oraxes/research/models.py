"""Validated domain models for research, evidence, and agent control."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceSearchResult(BaseModel):
    """One result returned by a web search provider."""

    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    snippet: str = Field(default="", max_length=2_000)


class PageSnapshot(BaseModel):
    """Readable, bounded representation of the current browser page."""

    url: HttpUrl
    title: str = Field(default="", max_length=500)
    text: str = Field(default="", max_length=20_000)


class PageElement(BaseModel):
    """A visible page element extracted through a CSS selector."""

    text: str = Field(default="", max_length=2_000)
    tag: str = Field(default="", max_length=100)
    href: str | None = Field(default=None, max_length=2_000)


class Citation(BaseModel):
    """A source quote that can be checked by the reader."""

    id: str = Field(pattern=r"^S[1-9][0-9]*$")
    url: HttpUrl
    title: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=1_500)
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Evidence(BaseModel):
    """A claim-relevant observation tied to one collected citation."""

    claim: str = Field(min_length=1, max_length=1_000)
    citation: Citation
    relevance: str = Field(min_length=1, max_length=1_000)


class ResearchFinding(BaseModel):
    """A concise, supported finding in the final report."""

    claim: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0, le=1)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("evidence_ids must be unique")
        return value


class ResearchResult(BaseModel):
    """The final structured, citation-backed response from the agent."""

    question: str = Field(min_length=1, max_length=4_000)
    answer: str = Field(min_length=1, max_length=10_000)
    findings: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    citations: list[Citation] = Field(default_factory=list, max_length=30)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("citations")
    @classmethod
    def unique_citation_ids(cls, value: list[Citation]) -> list[Citation]:
        identifiers = [citation.id for citation in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("citation IDs must be unique")
        return value


class ToolCall(BaseModel):
    """A provider-produced, structured request to execute a local tool."""

    id: str = Field(min_length=1, max_length=200)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The serialized output returned to a model after executing a tool."""

    tool_call_id: str
    name: str
    content: dict[str, object]
    is_error: bool = False


class AgentEventKind(StrEnum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPLETE = "complete"


class AgentEvent(BaseModel):
    """An inspectable in-memory trace entry for one agent-loop event."""

    kind: AgentEventKind
    detail: str
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
