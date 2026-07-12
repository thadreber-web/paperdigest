import pytest

from paperdigest import llm
from paperdigest.llm import (
    AnthropicBackend,
    LLMError,
    OpenAICompatibleBackend,
    complete_with_retry,
    make_backend,
)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_backend("anthropic", "claude-sonnet-5")


def test_openai_requires_key_without_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_backend("openai", "gpt-5")


def test_local_server_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = make_backend("openai", "llama3.1", base_url="http://localhost:11434/v1")
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.model == "llama3.1"


def test_local_backend_needs_no_key_and_defaults_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = make_backend("local", "local")
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.base_url == "http://localhost:8080/v1"
    custom = make_backend("local", "qwen35-9b", base_url="http://localhost:8001/v1")
    assert custom.base_url == "http://localhost:8001/v1"


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(make_backend("anthropic", "m"), AnthropicBackend)
    with pytest.raises(LLMError, match="unknown backend"):
        make_backend("mistral", "m")


class _Flaky:
    model = "flaky"

    def __init__(self, fail_times):
        self.remaining_failures = fail_times

    def complete(self, system, user):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("transient")
        return "ok"


def test_retry_recovers():
    assert complete_with_retry(_Flaky(2), "s", "u", retries=2) == "ok"


def test_retry_exhausted_raises_llmerror():
    with pytest.raises(LLMError, match="after 3 attempts"):
        complete_with_retry(_Flaky(99), "s", "u", retries=2)


from paperdigest.llm import strip_fences


def test_strip_fences_removes_markdown_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_passes_plain_text_through():
    assert strip_fences('  {"a": 1}\n') == '{"a": 1}'
