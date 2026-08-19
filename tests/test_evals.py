import json
from pathlib import Path


def test_agent_eval_cases_cover_the_critical_evidence_workflow() -> None:
    cases = json.loads((Path(__file__).parents[1] / "evals" / "cases.json").read_text())
    known = {
        "search",
        "fetch_page",
        "browser_open",
        "browser_click",
        "browser_extract",
        "collect_evidence",
        "complete_research",
    }
    assert len(cases) >= 2
    assert all(set(case["required_tools"]).issubset(known) for case in cases)
