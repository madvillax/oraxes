from oraxes.tools.search import SearchTool


async def test_search_tool_parses_bounded_results() -> None:
    async def html(_: str) -> str:
        return """
        <article class='result'><a class='result__a' href='https://example.com/a'>First</a>
        <a class='result__snippet'>First snippet</a></article>
        <article class='result'><a class='result__a' href='https://example.com/b'>Second</a></article>
        """

    results = await SearchTool(client=html).run("test query", max_results=1)

    assert len(results) == 1
    assert results[0].title == "First"
    assert str(results[0].url) == "https://example.com/a"


async def test_search_tool_resolves_duckduckgo_redirect_urls() -> None:
    async def html(_: str) -> str:
        return """
        <article class='result'>
          <a class='result__a'
             href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fsuperplexer.com%2F'>Superplexer</a>
        </article>
        """

    results = await SearchTool(client=html).run("superplexer.com")

    assert str(results[0].url) == "https://superplexer.com/"
