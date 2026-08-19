"""Execution boundary between LLM-generated tool calls and local capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from oraxes.agent.state import AgentState
from oraxes.browser.session import BrowserSession
from oraxes.research.models import (
    AgentEvent,
    AgentEventKind,
    Citation,
    Evidence,
    PageSnapshot,
    ToolCall,
    ToolResult,
)
from oraxes.tools.fetch import PageFetcher
from oraxes.tools.schemas import (
    BackArgs,
    ClickArgs,
    CollectEvidenceArgs,
    CompleteResearchArgs,
    ExtractElementsArgs,
    FetchPageArgs,
    NavigateArgs,
    ScrollArgs,
    SearchArgs,
    TypeArgs,
)
from oraxes.tools.search import SearchTool


class ToolDefinition(BaseModel):
    """A provider-neutral JSON-schema function declaration."""

    name: str
    description: str
    parameters: dict[str, object]


ToolHandler = Callable[[BaseModel], Awaitable[dict[str, object]]]


class ToolRegistry:
    """Validates every tool argument before dispatching it to an async handler."""

    def __init__(
        self,
        browser: BrowserSession,
        search: SearchTool | None = None,
        fetcher: PageFetcher | None = None,
    ) -> None:
        self._browser = browser
        self._search = search or SearchTool()
        self._fetcher = fetcher or PageFetcher()
        self._current_snapshot: PageSnapshot | None = None
        self._definitions: dict[str, tuple[str, type[BaseModel], ToolHandler]] = {}
        self._register_defaults()

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=name, description=description, parameters=schema.model_json_schema()
            )
            for name, (description, schema, _) in self._definitions.items()
        ]

    def _register(
        self, name: str, description: str, schema: type[BaseModel], handler: ToolHandler
    ) -> None:
        self._definitions[name] = (description, schema, handler)

    def _register_defaults(self) -> None:
        self._register("search", self._search.description, SearchArgs, self._search_web)
        self._register(
            "fetch_page",
            "Fetch and extract a static page quickly without launching a browser.",
            FetchPageArgs,
            self._fetch_page,
        )
        self._register(
            "browser_open",
            "Render an absolute URL in Chromium when fetch_page is insufficient.",
            NavigateArgs,
            self._navigate,
        )
        self._register(
            "browser_click",
            "Click the first rendered element matching a CSS selector.",
            ClickArgs,
            self._click,
        )
        self._register(
            "browser_type",
            "Fill a rendered text field selected by CSS selector.",
            TypeArgs,
            self._type,
        )
        self._register(
            "browser_extract",
            "Extract visible text or selected elements from the rendered page.",
            ExtractElementsArgs,
            self._extract_elements,
        )
        self._register(
            "browser_scroll",
            "Scroll the rendered page vertically by pixels.",
            ScrollArgs,
            self._scroll,
        )
        self._register(
            "browser_back", "Return to the previous rendered page.", BackArgs, self._back
        )
        self._register(
            "collect_evidence",
            "Save a direct quote from the current page as citable evidence.",
            CollectEvidenceArgs,
            self._collect_evidence,
        )
        self._register(
            "complete_research",
            "Request final synthesis after sufficient evidence.",
            CompleteResearchArgs,
            self._complete,
        )

    async def execute(self, call: ToolCall, state: AgentState) -> ToolResult:
        definition = self._definitions.get(call.name)
        if definition is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content={"error": f"Unknown tool: {call.name}"},
                is_error=True,
            )
        _, schema, handler = definition
        try:
            arguments = schema.model_validate(call.arguments)
            content = await handler(arguments)
            if call.name == "collect_evidence":
                evidence = content.pop("evidence")
                if not isinstance(evidence, Evidence):
                    raise RuntimeError("collect_evidence returned invalid evidence")
                citation = evidence.citation.model_copy(
                    update={"id": f"S{len(state.evidence) + 1}"}
                )
                state.add_evidence(evidence.model_copy(update={"citation": citation}))
                content["citation_id"] = citation.id
            if call.name == "complete_research":
                state.completion_requested = True
            result = ToolResult(tool_call_id=call.id, name=call.name, content=content)
        except Exception as error:
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content={"error": str(error)},
                is_error=True,
            )
        state.tool_results.append(result)
        state.events.append(
            AgentEvent(kind=AgentEventKind.TOOL_RESULT, detail=f"{call.name}: {result.is_error}")
        )
        return result

    async def _search_web(self, args: BaseModel) -> dict[str, object]:
        parsed = SearchArgs.model_validate(args)
        results = await self._search.run(parsed.query, parsed.max_results)
        return {"results": [result.model_dump(mode="json") for result in results]}

    async def _fetch_page(self, args: BaseModel) -> dict[str, object]:
        parsed = FetchPageArgs.model_validate(args)
        self._current_snapshot = await self._fetcher.fetch(str(parsed.url), parsed.max_chars)
        return self._current_snapshot.model_dump(mode="json")

    async def _navigate(self, args: BaseModel) -> dict[str, object]:
        self._current_snapshot = await self._browser.navigate(
            str(NavigateArgs.model_validate(args).url)
        )
        return self._current_snapshot.model_dump(mode="json")

    async def _click(self, args: BaseModel) -> dict[str, object]:
        self._current_snapshot = await self._browser.click(ClickArgs.model_validate(args).selector)
        return self._current_snapshot.model_dump(mode="json")

    async def _type(self, args: BaseModel) -> dict[str, object]:
        parsed = TypeArgs.model_validate(args)
        await self._browser.type(parsed.selector, parsed.text)
        return {"status": "typed"}

    async def _extract_elements(self, args: BaseModel) -> dict[str, object]:
        parsed = ExtractElementsArgs.model_validate(args)
        self._current_snapshot = await self._browser.read_page(parsed.max_chars)
        if parsed.selector == "body":
            return self._current_snapshot.model_dump(mode="json")
        elements = await self._browser.extract_elements(parsed.selector, parsed.limit)
        return {
            "url": str(self._current_snapshot.url),
            "title": self._current_snapshot.title,
            "elements": [element.model_dump(mode="json") for element in elements],
        }

    async def _scroll(self, args: BaseModel) -> dict[str, object]:
        self._current_snapshot = await self._browser.scroll(ScrollArgs.model_validate(args).pixels)
        return self._current_snapshot.model_dump(mode="json")

    async def _back(self, _: BaseModel) -> dict[str, object]:
        self._current_snapshot = await self._browser.back()
        return self._current_snapshot.model_dump(mode="json")

    async def _collect_evidence(self, args: BaseModel) -> dict[str, object]:
        parsed = CollectEvidenceArgs.model_validate(args)
        snapshot = self._current_snapshot
        if snapshot is None:
            raise ValueError("Use fetch_page or browser_extract before collecting evidence")
        normalized_page = " ".join(snapshot.text.split())
        normalized_quote = " ".join(parsed.quote.split())
        if normalized_quote not in normalized_page:
            raise ValueError("Evidence quote must appear verbatim in the current page text")
        evidence = {
            "claim": parsed.claim,
            "relevance": parsed.relevance,
            "citation": Citation(
                id="S1",
                url=snapshot.url,
                title=snapshot.title or str(snapshot.url),
                quote=parsed.quote,
            ),
        }
        return {"evidence": Evidence.model_validate(evidence)}

    async def _complete(self, args: BaseModel) -> dict[str, object]:
        return {
            "status": "completion_requested",
            "reason": CompleteResearchArgs.model_validate(args).reason,
        }
