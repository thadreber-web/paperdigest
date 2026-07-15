import threading
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
        self.json_modes = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        self.json_modes.append(json_mode)
        if not self.responses:
            raise AssertionError("FakeBackend ran out of responses")
        return self.responses.pop(0)


class KeyedFakeBackend:
    """Thread-safe fake backend for parallel tests: looks up a response by a marker
    substring found in the user prompt, rather than popping a shared queue in order."""

    model = "fake-model"

    def __init__(self, keyed_responses: dict[str, str], default: str | None = None):
        self.keyed_responses = keyed_responses
        self.default = default
        self._lock = threading.Lock()
        self.calls = []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        with self._lock:
            self.calls.append((system, user))
        for key, response in self.keyed_responses.items():
            if key in user:
                return response
        if self.default is not None:
            return self.default
        raise AssertionError(f"KeyedFakeBackend has no response matching user prompt: {user[:200]!r}")
