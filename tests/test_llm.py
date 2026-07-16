from types import SimpleNamespace

import httpx
import pytest
from conftest import FakeBackend

from paperdigest import llm
from paperdigest.llm import (
    AnthropicBackend,
    LLMError,
    OpenAICompatibleBackend,
    VisionUnsupportedError,
    complete_with_retry,
    make_backend,
    run_tasks,
    strip_fences,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16


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
            raise httpx.ConnectError("transient")
        return "ok"


def test_retry_recovers():
    assert complete_with_retry(_Flaky(2), "s", "u", retries=2) == "ok"


def test_retry_exhausted_raises_llmerror():
    with pytest.raises(LLMError, match="after 3 attempts"):
        complete_with_retry(_Flaky(99), "s", "u", retries=2)


def test_strip_fences_removes_markdown_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_passes_plain_text_through():
    assert strip_fences('  {"a": 1}\n') == '{"a": 1}'


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


def test_openai_backend_sends_configured_max_tokens():
    stub = _StubCompletions()
    backend = OpenAICompatibleBackend("m", base_url="http://localhost:9999/v1", max_tokens=2048)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    backend.complete("s", "u")
    assert stub.kwargs_seen[0]["max_tokens"] == 2048


def test_anthropic_backend_sends_configured_max_tokens(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5", max_tokens=2048)

    class _Messages:
        def create(self, **kwargs):
            self.kwargs_seen = kwargs
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="ok")],
            )

    stub = _Messages()
    backend._client = SimpleNamespace(messages=stub)
    backend.complete("s", "u")
    assert stub.kwargs_seen["max_tokens"] == 2048


def test_openai_backend_falls_back_when_server_rejects_response_format():
    stub = _StubCompletions(reject_response_format=True)
    backend = _openai_backend(stub)
    assert backend.complete("s", "u", json_mode=True) == "ok"
    assert "response_format" not in stub.kwargs_seen[-1]
    backend.complete("s", "u", json_mode=True)  # sticky: no second rejection round-trip
    assert len(stub.kwargs_seen) == 3
    assert "response_format" not in stub.kwargs_seen[-1]


def test_openai_backend_raises_on_truncation():
    class _TruncatedCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="partial..."),
                    )
                ]
            )

    backend = _openai_backend(_TruncatedCompletions())
    with pytest.raises(LLMError, match="truncated"):
        backend.complete("s", "u")


def test_anthropic_backend_raises_on_truncation(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5")

    class _TruncatedMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(type="text", text="partial...")],
            )

    backend._client = SimpleNamespace(messages=_TruncatedMessages())
    with pytest.raises(LLMError, match="truncated"):
        backend.complete("s", "u")


class _AlwaysFails:
    model = "broken"

    def __init__(self, exc):
        self.exc = exc
        self.attempts = 0

    def complete(self, system, user, json_mode=False):
        self.attempts += 1
        raise self.exc


def test_non_transient_error_fails_fast_without_retrying():
    backend = _AlwaysFails(ValueError("bad request, will never work"))
    with pytest.raises(LLMError, match="bad request"):
        complete_with_retry(backend, "s", "u", retries=2)
    assert backend.attempts == 1


def test_truncation_error_fails_fast_without_retrying():
    backend = _AlwaysFails(LLMError("truncated at max_tokens"))
    with pytest.raises(LLMError, match="truncated"):
        complete_with_retry(backend, "s", "u", retries=2)
    assert backend.attempts == 1


def test_transient_error_is_retried():
    backend = _Flaky(2)
    assert complete_with_retry(backend, "s", "u", retries=2) == "ok"


def test_strip_fences_handles_crlf():
    assert strip_fences('```json \r\n{"a": 1}\r\n```') == '{"a": 1}'
    assert strip_fences('```\r\n{"a": 1}\r\n```  ') == '{"a": 1}'


def test_run_tasks_serial_preserves_order():
    results = run_tasks([1, 2, 3], lambda x: x * 10, workers=1)
    assert results == [10, 20, 30]


def test_run_tasks_serial_propagates_error():
    def fn(x):
        if x == 2:
            raise LLMError("boom")
        return x

    with pytest.raises(LLMError, match="boom"):
        run_tasks([1, 2, 3], fn, workers=1)


def test_run_tasks_parallel_preserves_order_regardless_of_completion_order():
    import time as time_mod

    def fn(x):
        time_mod.sleep(0.02 if x == 1 else 0)  # first item finishes last
        return x * 10

    results = run_tasks([1, 2, 3, 4], fn, workers=4)
    assert results == [10, 20, 30, 40]


def test_run_tasks_parallel_raises_on_first_failure():
    def fn(x):
        if x == 3:
            raise LLMError("worker failed")
        return x

    with pytest.raises(LLMError, match="worker failed"):
        run_tasks([1, 2, 3, 4], fn, workers=4)


def test_openai_backend_sends_png_data_url_image():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    backend.complete("s", "u", images=[PNG_BYTES])
    content = stub.kwargs_seen[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "u"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_openai_backend_sends_jpeg_data_url_image():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    backend.complete("s", "u", images=[JPEG_BYTES])
    content = stub.kwargs_seen[0]["messages"][1]["content"]
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_openai_backend_multiple_images_all_present():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    backend.complete("s", "u", images=[PNG_BYTES, JPEG_BYTES])
    content = stub.kwargs_seen[0]["messages"][1]["content"]
    assert len(content) == 3  # text + 2 images
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_backend_no_images_keeps_plain_string_content():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    backend.complete("s", "u")
    assert stub.kwargs_seen[0]["messages"][1]["content"] == "u"


def test_openai_backend_json_mode_and_images_combined_does_not_crash():
    stub = _StubCompletions()
    backend = _openai_backend(stub)
    assert backend.complete("s", "u", json_mode=True, images=[PNG_BYTES]) == "ok"
    content = stub.kwargs_seen[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "u"}


def test_openai_backend_badrequest_without_images_keeps_existing_behavior():
    stub = _StubCompletions(reject_response_format=True)
    backend = _openai_backend(stub)
    # json_mode BadRequestError still degrades gracefully, no VisionUnsupportedError
    assert backend.complete("s", "u", json_mode=True) == "ok"
    assert "response_format" not in stub.kwargs_seen[-1]


def test_openai_backend_badrequest_with_images_raises_vision_unsupported():
    class _RejectingCompletions:
        def create(self, **kwargs):
            import openai

            req = httpx.Request("POST", "http://localhost/v1")
            raise openai.BadRequestError(
                "unsupported content type",
                response=httpx.Response(400, request=req),
                body=None,
            )

    backend = OpenAICompatibleBackend("m", base_url="http://localhost:9999/v1")
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=_RejectingCompletions()))
    with pytest.raises(VisionUnsupportedError):
        backend.complete("s", "u", images=[PNG_BYTES])


def test_openai_backend_badrequest_without_images_raises_plain_error():
    class _RejectingCompletions:
        def create(self, **kwargs):
            import openai

            req = httpx.Request("POST", "http://localhost/v1")
            raise openai.BadRequestError(
                "bad request", response=httpx.Response(400, request=req), body=None
            )

    backend = OpenAICompatibleBackend("m", base_url="http://localhost:9999/v1")
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=_RejectingCompletions()))
    import openai

    with pytest.raises(openai.BadRequestError):
        backend.complete("s", "u")


def test_anthropic_backend_sends_image_blocks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5")

    class _Messages:
        def create(self, **kwargs):
            self.kwargs_seen = kwargs
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="ok")],
            )

    stub = _Messages()
    backend._client = SimpleNamespace(messages=stub)
    backend.complete("s", "u", images=[PNG_BYTES, JPEG_BYTES])
    import base64

    content = stub.kwargs_seen["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == base64.b64encode(PNG_BYTES).decode("ascii")
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert content[2] == {"type": "text", "text": "u"}


def test_anthropic_backend_no_images_keeps_plain_string_content(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5")

    class _Messages:
        def create(self, **kwargs):
            self.kwargs_seen = kwargs
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text="ok")],
            )

    stub = _Messages()
    backend._client = SimpleNamespace(messages=stub)
    backend.complete("s", "u")
    assert stub.kwargs_seen["messages"][0]["content"] == "u"


def test_anthropic_backend_badrequest_with_images_raises_vision_unsupported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5")

    class _RejectingMessages:
        def create(self, **kwargs):
            import anthropic

            req = httpx.Request("POST", "http://localhost/v1")
            raise anthropic.BadRequestError(
                "unsupported content", response=httpx.Response(400, request=req), body=None
            )

    backend._client = SimpleNamespace(messages=_RejectingMessages())
    with pytest.raises(VisionUnsupportedError):
        backend.complete("s", "u", images=[PNG_BYTES])


def test_anthropic_backend_badrequest_without_images_raises_plain_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    backend = AnthropicBackend("claude-sonnet-5")

    class _RejectingMessages:
        def create(self, **kwargs):
            import anthropic

            req = httpx.Request("POST", "http://localhost/v1")
            raise anthropic.BadRequestError(
                "bad request", response=httpx.Response(400, request=req), body=None
            )

    backend._client = SimpleNamespace(messages=_RejectingMessages())
    import anthropic

    with pytest.raises(anthropic.BadRequestError):
        backend.complete("s", "u")


def test_complete_with_retry_passes_images_through():
    backend = FakeBackend(["ok"])
    complete_with_retry(backend, "s", "u", images=[PNG_BYTES])
    assert backend.images_calls == [[PNG_BYTES]]


def test_complete_with_retry_without_images_uses_legacy_signature():
    """Backends without an images param in their signature must keep working."""

    class _LegacyBackend:
        model = "legacy"

        def complete(self, system, user, json_mode=False):
            return "ok"

    assert complete_with_retry(_LegacyBackend(), "s", "u") == "ok"


def test_complete_with_retry_vision_unsupported_not_retried():
    class _AlwaysVisionRejects:
        model = "rejector"

        def __init__(self):
            self.attempts = 0

        def complete(self, system, user, json_mode=False, images=None):
            self.attempts += 1
            raise VisionUnsupportedError("no vision support")

    backend = _AlwaysVisionRejects()
    with pytest.raises(VisionUnsupportedError):
        complete_with_retry(backend, "s", "u", retries=2, images=[PNG_BYTES])
    assert backend.attempts == 1
