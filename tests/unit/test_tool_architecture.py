from oraxes.browser.session import BrowserSession
from oraxes.tools.registry import ToolRegistry


def test_registry_exposes_fast_fetch_first_architecture() -> None:
    names = {definition.name for definition in ToolRegistry(BrowserSession()).definitions}

    assert {"search", "fetch_page", "browser_open", "browser_click", "browser_extract"} <= names
    assert {"search_web", "navigate", "read_page", "extract_elements"}.isdisjoint(names)
