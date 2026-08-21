# Oraxes

Oraxes is a terminal-operated web research agent. Give it a question and it uses Gemini to plan a bounded sequence of web-search and Playwright browser actions, collects verbatim evidence, and prints a structured report with source citations.

## V1

- Async-first custom agent loop—no LangChain, LangGraph, Browser Use, or multi-agent framework.
- Gemini behind a small `LLMProvider` interface, making another provider straightforward to add.
- Structured Gemini function calls and a Pydantic-constrained structured final result.
- Fast tools: `search` and `fetch_page` retrieve static evidence without starting Chromium.
- Rendered tools: `browser_open`, `browser_click`, and `browser_extract`, plus interactive type, scroll, and back helpers.
- Research tools: DuckDuckGo HTML search, direct evidence collection, source citations, and explicit completion.
- In-memory state only; no database, API server, telemetry platform, or persistence.
- Unit, integration, browser E2E, and agent-eval fixture coverage.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key

Install dependencies and Chromium:

```bash
uv sync --dev
uv run playwright install chromium
```

Configure the provider:

```bash
cp .env.example .env
export GEMINI_API_KEY="your-key"
# optional: export ORAXES_MODEL="gemini-3.5-flash-lite"
```

Oraxes loads `.env` automatically. You can also export `GEMINI_API_KEY` in the shell; an exported value takes precedence.

## Run research

```bash
uv run oraxes "What are the latest official accessibility requirements for public websites?"
```

Useful options:

```bash
uv run oraxes "Compare two primary sources" --max-steps 16
uv run oraxes "Inspect this interactive site" --headed
```

The terminal output contains the answer, supported findings, evidence IDs, direct source quotes, URLs, and limitations. A model cannot create a citation by itself: `collect_evidence` validates its quote against the latest fetched or rendered page before the source enters state.

## Architecture

```text
Typer CLI
   -> ResearchAgent (bounded asyncio loop)
       -> LLMProvider -> GeminiProvider (function calls + JSON final output)
       -> ToolRegistry (Pydantic argument validation)
           -> SearchTool + PageFetcher (fast static path)
           -> BrowserSession (lazy async Playwright fallback)
           -> Evidence and citation state (memory only)
```

The LLM provider makes decisions; application code validates arguments, executes tools serially (one browser tab is intentionally stateful), rejects invalid evidence, and owns completion. This keeps action execution auditable and provider-independent.

The default fast path is `search` → `fetch_page` → `collect_evidence`. Chromium is launched lazily only for the fallback path: `browser_open` → `browser_click` → `browser_extract`. This avoids browser startup and JavaScript rendering for ordinary static sources.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
```

Test layers:

- `tests/unit`: Pydantic validation and search-result parsing.
- `tests/integration`: scripted provider + deterministic browser, exercising the full loop and evidence gate.
- `tests/e2e`: real async Playwright against a local HTTP server; no external website dependency.
- `evals/cases.json`: lightweight behavioral cases that protect the evidence-first contract.

## Layout

```text
src/oraxes/
  cli.py                 terminal entry point and Rich rendering
  agent/                 state and custom research loop
  llm/                   provider protocol and Gemini adapter
  browser/               async Playwright session
  tools/                 search, schemas, and validated dispatcher
  research/              Pydantic output models
tests/                   unit, integration, and browser E2E tests
evals/                   agent evaluation cases
```
