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

    def complete(self, system, user, json_mode=False):
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


from types import SimpleNamespace

import httpx

from conftest import FakeBackend
from paperdigest.llm import OpenAICompatibleBackend


def test_complete_with_retry_passes_json_mode():
    backend = FakeBackend(["{}"])
    complete_with_retry(backend, "s", "u", json_mode=True)
    assert backend.json_modes == [True]


class _StubCompletions:
    def __init__(self, reject_response_format=False):
        self.kwargs_seen = []
        self.reject_response_format = reject_response_format

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        if self.reject_response_format and "response_format" in kwargs:
            import openai

            req = httpx.Request("POST", "http://localhost/v1")
            raise openai.BadRequestError(
                "response_format unsupported", response=httpx.Response(400, request=req), body=None
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def _openai_backend(stub):
    backend = OpenAICompatibleBackend("m", base_url="http://localhost:9999/v1")
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return backend


def test_openai_backend_requests_json_object_when_json_mode():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    assert backend.complete("s", "u", json_mode=True) == "ok"
    assert stub.kwargs_seen[0]["response_format"] == {"type": "json_object"}


def test_openai_backend_omits_response_format_by_default():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    backend.complete("s", "u")
    assert "response_format" not in stub.kwargs_seen[0]


def test_openai_backend_falls_back_when_server_rejects_response_format():
    stub = _StubCompletions(reject_response_format=True)
    backend = _openai_backend(stub)
    assert backend.complete("s", "u", json_mode=True) == "ok"
    assert "response_format" not in stub.kwargs_seen[-1]
    backend.complete("s", "u", json_mode=True)  # sticky: no second rejection round-trip
    assert len(stub.kwargs_seen) == 3
    assert "response_format" not in stub.kwargs_seen[-1]
