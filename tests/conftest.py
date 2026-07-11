from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_html() -> str:
    return (FIXTURES / "paper.html").read_text()


class FakeBackend:
    """Queue-based fake LLM backend; returns responses in order."""

    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("FakeBackend ran out of responses")
        return self.responses.pop(0)
