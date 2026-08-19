"""Gemini implementation using structured function calls and JSON output."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, cast

from google import genai
from pydantic import BaseModel, Field

from oraxes.agent.state import AgentState
from oraxes.research.models import ResearchFinding, ResearchResult, ToolCall
from oraxes.tools.registry import ToolDefinition


@dataclass(frozen=True)
class GeminiTurn:
    tool_calls: list[ToolCall]
    text: str


class GeminiConfigurationError(RuntimeError):
    """A user-actionable Gemini credential or configuration error."""


class GeminiRateLimitError(RuntimeError):
    """Gemini quota remained unavailable after bounded retries."""


class SynthesisFinding(BaseModel):
    """Provider-facing finding kept within Gemini's JSON Schema subset."""

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float


class SynthesisOutput(BaseModel):
    """Provider-facing output; trusted citations are attached by the application."""

    answer: str
    findings: list[SynthesisFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class GeminiProvider:
    """Google GenAI adapter. It retains only one in-memory interaction history per run."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing. Add a valid key to .env or export it in your shell."
            )
        if "replace-with" in key.lower() or key.lower() == "your-key":
            raise GeminiConfigurationError(
                "GEMINI_API_KEY still contains the example placeholder. "
                "Replace it with a real Gemini API key."
            )
        self._client = genai.Client(api_key=key)
        self._model = model or os.getenv("ORAXES_MODEL", "gemini-3.5-flash-lite")
        self._history: list[dict[str, Any]] = []
        self._returned_results = 0

    async def decide(
        self, question: str, state: AgentState, tools: list[ToolDefinition]
    ) -> GeminiTurn:
        if not self._history:
            self._history.append(
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": self._planner_prompt(question),
                        }
                    ],
                }
            )
        self._append_tool_results(state)
        declarations = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
        interaction = await self._create_interaction(
            model=self._model,
            input=self._history,
            tools=declarations,
            store=False,
        )
        steps = [step.model_dump(exclude_none=True) for step in (interaction.steps or [])]
        self._history.extend(steps)
        tool_calls = [
            ToolCall(id=step.id, name=step.name, arguments=dict(step.arguments or {}))
            for step in (interaction.steps or [])
            if step.type == "function_call"
        ]
        return GeminiTurn(tool_calls=tool_calls, text=interaction.output_text or "")

    async def synthesize(self, question: str, state: AgentState) -> ResearchResult:
        evidence_json = json.dumps(
            [evidence.model_dump(mode="json") for evidence in state.evidence], indent=2
        )
        prompt = f"""Answer this research question: {question}

Use only the collected evidence below. Every finding must reference one or more
existing citation IDs. Keep uncertainty explicit. Do not invent a source, quote,
or URL. If evidence is insufficient, say so in limitations.

Collected evidence:
{evidence_json}
"""
        interaction = await self._create_interaction(
            model=self._model,
            input=prompt,
            store=False,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SynthesisOutput.model_json_schema(),
            },
        )
        if not interaction.output_text:
            raise RuntimeError("Gemini returned no structured research output")
        synthesis = SynthesisOutput.model_validate_json(interaction.output_text)
        result = ResearchResult(
            question=question,
            answer=synthesis.answer,
            findings=[
                ResearchFinding(
                    claim=finding.claim,
                    evidence_ids=finding.evidence_ids,
                    confidence=max(0.0, min(1.0, finding.confidence)),
                )
                for finding in synthesis.findings
                if finding.evidence_ids
            ],
            limitations=synthesis.limitations,
        )
        allowed = {citation.id for citation in state.citations}
        result.citations = [citation for citation in result.citations if citation.id in allowed]
        for finding in result.findings:
            finding.evidence_ids = [
                identifier for identifier in finding.evidence_ids if identifier in allowed
            ]
        result.findings = [finding for finding in result.findings if finding.evidence_ids]
        return result

    @staticmethod
    def _planner_prompt(question: str) -> str:
        return f"""You are a careful web research planner. Research: {question}

Use this fast path: search, then fetch_page for credible static sources, then
collect_evidence with verbatim quotes. Only use browser_open, browser_click, and
browser_extract when fetch_page cannot expose the needed content. The optional
browser_type, browser_scroll, and browser_back tools handle interactive pages.
Collect at least two independent sources where the question permits. Do not
answer from memory. Call complete_research only after sufficient direct evidence
exists. For an explicit domain or URL, fetch_page it before search. Prefer primary
sources and preserve uncertainty."""

    def _append_tool_results(self, state: AgentState) -> None:
        for result in state.tool_results[self._returned_results :]:
            self._history.append(
                {
                    "type": "function_result",
                    "name": result.name,
                    "call_id": result.tool_call_id,
                    "result": [{"type": "text", "text": json.dumps(result.content)}],
                }
            )
        self._returned_results = len(state.tool_results)

    async def _create_interaction(self, **kwargs: object) -> Any:
        interactions = cast(Any, self._client.aio.interactions)
        for attempt in range(4):
            try:
                return await interactions.create(**kwargs)
            except Exception as error:
                if self._is_rate_limit_error(error) and attempt < 3:
                    await asyncio.sleep(self._retry_delay_seconds(error, attempt))
                    continue
                if self._is_rate_limit_error(error):
                    raise GeminiRateLimitError(
                        "Gemini quota is still unavailable after 3 retries. Wait for the "
                        "quota window to reset, reduce --max-steps, or check Gemini billing."
                    ) from error
                if translated := self._translate_request_error(error):
                    raise translated from error
                raise
        raise RuntimeError("Gemini retry loop exited unexpectedly")

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return "too_many_requests" in message or "exceeded your current quota" in message

    @staticmethod
    def _retry_delay_seconds(error: Exception, attempt: int) -> float:
        match = re.search(r"retry in ([0-9.]+)s", str(error), flags=re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 0.5, 15.0)
        return min(float(2**attempt), 8.0)

    def _translate_request_error(self, error: Exception) -> GeminiConfigurationError | None:
        message = str(error)
        if "API_KEY_INVALID" in message or "API key not valid" in message:
            return GeminiConfigurationError(
                "Gemini rejected GEMINI_API_KEY. Create or copy a valid Gemini API key, "
                "then update .env or your shell environment and retry."
            )
        if "Multimodal function responses are not supported" in message:
            return GeminiConfigurationError(
                f"{self._model} cannot continue this browser-tool loop. "
                "Set ORAXES_MODEL=gemini-3.5-flash-lite and retry."
            )
        return None
