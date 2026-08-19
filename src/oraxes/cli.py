"""Terminal interface for the Oraxes research agent."""

from __future__ import annotations

import asyncio

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from oraxes.agent.researcher import ResearchAgent
from oraxes.browser.session import BrowserSession
from oraxes.llm.gemini import GeminiConfigurationError, GeminiProvider, GeminiRateLimitError
from oraxes.research.models import ResearchResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Citation-backed web research from your terminal.",
)
console = Console()


def render_result(result: ResearchResult) -> None:
    console.print(Panel.fit(result.question, title="Research question", border_style="cyan"))
    console.print(Markdown(result.answer))
    if result.findings:
        findings = Table(title="Findings", show_header=True, header_style="bold cyan")
        findings.add_column("Claim", ratio=4)
        findings.add_column("Evidence", ratio=1)
        findings.add_column("Confidence", ratio=1)
        for finding in result.findings:
            findings.add_row(
                finding.claim, ", ".join(finding.evidence_ids), f"{finding.confidence:.0%}"
            )
        console.print(findings)
    if result.citations:
        citations = Table(title="Sources", show_header=True, header_style="bold cyan")
        citations.add_column("ID", style="bold")
        citations.add_column("Source", ratio=2)
        citations.add_column("Supporting quote", ratio=4)
        for citation in result.citations:
            citations.add_row(citation.id, f"{citation.title}\n{citation.url}", citation.quote)
        console.print(citations)
    if result.limitations:
        console.print(
            Panel(
                "\n".join(f"• {item}" for item in result.limitations),
                title="Limitations",
                border_style="yellow",
            )
        )


@app.command()
def research(
    question: str = typer.Argument(..., help="The question to investigate."),
    max_steps: int = typer.Option(12, min=1, max=30, help="Maximum planning/tool turns."),
    headed: bool = typer.Option(False, help="Show the browser while researching."),
) -> None:
    """Research a question using Gemini, web search, and a controlled browser."""
    load_dotenv()
    try:
        provider = GeminiProvider()
        agent = ResearchAgent(
            provider,
            max_steps=max_steps,
            browser_factory=lambda: BrowserSession(headless=not headed),
        )
        with console.status("Researching sources and collecting evidence…", spinner="dots"):
            result = asyncio.run(agent.run(question))
    except GeminiConfigurationError as error:
        console.print(Panel(str(error), title="Gemini configuration error", border_style="red"))
        raise typer.Exit(code=2) from error
    except GeminiRateLimitError as error:
        console.print(Panel(str(error), title="Gemini rate limit", border_style="yellow"))
        raise typer.Exit(code=3) from error
    render_result(result)


if __name__ == "__main__":
    app()
