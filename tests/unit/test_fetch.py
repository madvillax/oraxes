import httpx

from oraxes.tools.fetch import PageFetcher


async def test_fetch_page_extracts_text_without_browser_or_scripts() -> None:
    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<title>Fast</title><script>ignore()</script><main>Useful evidence.</main>",
        )

    snapshot = await PageFetcher(httpx.MockTransport(response)).fetch("https://example.com")

    assert snapshot.title == "Fast"
    assert snapshot.text == "Fast\nUseful evidence."
    assert "ignore" not in snapshot.text
