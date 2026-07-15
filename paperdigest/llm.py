from __future__ import annotations

import os
import re
import time
from typing import Protocol


class LLMError(Exception):
    pass


LOCAL_BASE_URL = "http://localhost:8080/v1"  # llama.cpp server default
REQUEST_TIMEOUT = 300.0  # local models are slow — a smoke test showed multi-minute stages


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*[ \t]*\r?\n", "", text)
        text = re.sub(r"\r?\n```\s*$", "", text)
    return text


class Backend(Protocol):
    model: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class AnthropicBackend:
    def __init__(self, model: str):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        # Anthropic has no response_format switch; the prompts already demand JSON.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise LLMError("Anthropic response truncated at max_tokens")
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAICompatibleBackend:
    def __init__(self, model: str, base_url: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None and not api_key:
            raise LLMError("OPENAI_API_KEY is not set (or pass --base-url for a local server)")
        import openai

        self.model = model
        self.base_url = base_url
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=REQUEST_TIMEOUT
        )
        self._json_mode_unsupported = False

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        import openai

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if json_mode and not self._json_mode_unsupported:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                return self._extract_content(resp)
            except openai.BadRequestError:
                # server build doesn't support response_format — degrade for the rest of the run
                self._json_mode_unsupported = True
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        return self._extract_content(resp)

    @staticmethod
    def _extract_content(resp) -> str:
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("OpenAI-compatible response truncated at max_tokens")
        return choice.message.content or ""


def make_backend(backend: str, model: str, base_url: str | None = None) -> Backend:
    if backend == "local":
        return OpenAICompatibleBackend(model, base_url or LOCAL_BASE_URL)
    if backend == "anthropic":
        return AnthropicBackend(model)
    if backend == "openai":
        return OpenAICompatibleBackend(model, base_url)
    raise LLMError(f"unknown backend: {backend!r}")


def _is_transient(e: Exception) -> bool:
    """True for connection/timeout/rate-limit/server errors worth retrying."""
    import httpx

    if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
        return True

    try:
        import anthropic

        if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError)):
            return True
        if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
            return True
    except ImportError:
        pass

    try:
        import openai

        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
            return True
        if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
            return True
    except ImportError:
        pass

    return False


def complete_with_retry(
    backend: Backend, system: str, user: str, retries: int = 2, json_mode: bool = False
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return backend.complete(system, user, json_mode=json_mode)
        except Exception as e:  # noqa: BLE001 — SDK exception types vary per backend
            last_error = e
            if not _is_transient(e):
                raise LLMError(f"LLM call failed: {e}") from e
            if attempt < retries:
                time.sleep(2**attempt)
    raise LLMError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error
