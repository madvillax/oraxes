from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from oraxes.browser.session import BrowserSession


class LocalPageHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/next":
            body = (
                "<html><title>Next</title><body><p id='fact'>Evidence is visible here."
                "</p></body></html>"
            )
        else:
            body = (
                "<html><title>Start</title><body><input id='query'><a href='/next'>Next</a>"
                "</body></html>"
            )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_: object) -> None:
        return None


@pytest.mark.e2e
async def test_browser_session_can_navigate_type_click_read_and_go_back() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalPageHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with BrowserSession() as browser:
            start = await browser.navigate(f"http://127.0.0.1:{server.server_port}/")
            assert start.title == "Start"
            await browser.type("#query", "research")
            next_page = await browser.click("a")
            assert "Evidence is visible here." in next_page.text
            elements = await browser.extract_elements("#fact")
            assert elements[0].text == "Evidence is visible here."
            back = await browser.back()
            assert back.title == "Start"
    finally:
        server.shutdown()
        thread.join()
