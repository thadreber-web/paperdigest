from __future__ import annotations

import os
import re
import time
from typing import Protocol


class LLMError(Exception):
    pass


LOCAL_BASE_URL = "http://localhost:8080/v1"  # llama.cpp server default


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
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
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        # Anthropic has no response_format switch; the prompts already demand JSON.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAICompatibleBackend:
    def __init__(self, model: str, base_url: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None and not api_key:
            raise LLMError("OPENAI_API_KEY is not set (or pass --base-url for a local server)")
        import openai

        self.model = model
        self.base_url = base_url
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed")
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
                return resp.choices[0].message.content or ""
            except openai.BadRequestError:
                # server build doesn't support response_format — degrade for the rest of the run
                self._json_mode_unsupported = True
        resp = self._client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""


def make_backend(backend: str, model: str, base_url: str | None = None) -> Backend:
    if backend == "local":
        return OpenAICompatibleBackend(model, base_url or LOCAL_BASE_URL)
    if backend == "anthropic":
        return AnthropicBackend(model)
    if backend == "openai":
        return OpenAICompatibleBackend(model, base_url)
    raise LLMError(f"unknown backend: {backend!r}")


def complete_with_retry(
    backend: Backend, system: str, user: str, retries: int = 2, json_mode: bool = False
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return backend.complete(system, user, json_mode=json_mode)
        except Exception as e:  # noqa: BLE001 — SDK exception types vary per backend
            last_error = e
            if attempt < retries:
                time.sleep(2**attempt)
    raise LLMError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error
