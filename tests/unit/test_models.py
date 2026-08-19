from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from oraxes.research.models import Citation, ResearchResult


def test_research_result_rejects_duplicate_citation_ids() -> None:
    citation = Citation(
        id="S1",
        url="https://example.com",
        title="Example",
        quote="A direct quote.",
        collected_at=datetime.now(UTC),
    )
    with pytest.raises(ValidationError, match="citation IDs must be unique"):
        ResearchResult(question="Question?", answer="Answer.", citations=[citation, citation])
