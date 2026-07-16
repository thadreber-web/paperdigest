from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Protocol, TypeVar


class LLMError(Exception):
    pass


class VisionUnsupportedError(LLMError):
    """Raised when a backend/server rejects a call that included images."""


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

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        images: list[bytes] | None = None,
    ) -> str: ...


def _image_media_type(image: bytes) -> str:
    """Sniff JPEG vs PNG from magic bytes; default to PNG."""
    if image[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/png"


class AnthropicBackend:
    def __init__(self, model: str, max_tokens: int = 8192):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT)

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        images: list[bytes] | None = None,
    ) -> str:
        import base64

        import anthropic

        # Anthropic has no response_format switch; the prompts already demand JSON.
        if images:
            content: list[dict] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _image_media_type(image),
                        "data": base64.b64encode(image).decode("ascii"),
                    },
                }
                for image in images
            ]
            content.append({"type": "text", "text": user})
        else:
            content = user

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.BadRequestError as e:
            if images:
                raise VisionUnsupportedError(f"Anthropic rejected image content: {e}") from e
            raise
        if getattr(resp, "stop_reason", None) == "max_tokens":
            raise LLMError("Anthropic response truncated at max_tokens")
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAICompatibleBackend:
    def __init__(self, model: str, base_url: str | None = None, max_tokens: int = 8192):
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None and not api_key:
            raise LLMError("OPENAI_API_KEY is not set (or pass --base-url for a local server)")
        import openai

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=REQUEST_TIMEOUT
        )
        self._json_mode_unsupported = False

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        images: list[bytes] | None = None,
    ) -> str:
        import base64

        import openai

        if images:
            content: list[dict] | str = [{"type": "text", "text": user}]
            for image in images:
                mime = _image_media_type(image)
                b64 = base64.b64encode(image).decode("ascii")
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
        else:
            content = user

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        if json_mode and not self._json_mode_unsupported:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_tokens,
                )
                return self._extract_content(resp)
            except openai.BadRequestError:
                # server build doesn't support response_format — degrade for the rest of the run
                self._json_mode_unsupported = True
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=self.max_tokens
            )
        except openai.BadRequestError as e:
            if images:
                raise VisionUnsupportedError(f"backend rejected image content: {e}") from e
            raise
        return self._extract_content(resp)

    @staticmethod
    def _extract_content(resp) -> str:
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("OpenAI-compatible response truncated at max_tokens")
        return choice.message.content or ""


def make_backend(
    backend: str, model: str, base_url: str | None = None, max_tokens: int = 8192
) -> Backend:
    if backend == "local":
        return OpenAICompatibleBackend(model, base_url or LOCAL_BASE_URL, max_tokens)
    if backend == "anthropic":
        return AnthropicBackend(model, max_tokens)
    if backend == "openai":
        return OpenAICompatibleBackend(model, base_url, max_tokens)
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
    backend: Backend,
    system: str,
    user: str,
    retries: int = 2,
    json_mode: bool = False,
    images: list[bytes] | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # images kwarg passed only when actually used, so callers with a minimal
            # (pre-vision) complete(self, system, user, json_mode=False) signature —
            # e.g. test fakes outside this module — keep working unchanged.
            if images is not None:
                return backend.complete(system, user, json_mode=json_mode, images=images)
            return backend.complete(system, user, json_mode=json_mode)
        except VisionUnsupportedError:
            raise  # not transient, and callers need the specific type to skip figures
        except Exception as e:  # noqa: BLE001 — SDK exception types vary per backend
            last_error = e
            if not _is_transient(e):
                raise LLMError(f"LLM call failed: {e}") from e
            if attempt < retries:
                time.sleep(2**attempt)
    raise LLMError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error


_REPAIR_SYSTEM = (
    "The user message was supposed to be valid JSON but is not. Return ONLY the corrected JSON, nothing else."
)


def repair_json(backend: Backend, raw: str) -> str:
    """One-shot repair round-trip: ask the model to fix JSON it just produced but that failed to parse."""
    return complete_with_retry(backend, _REPAIR_SYSTEM, raw, json_mode=True)


T = TypeVar("T")
R = TypeVar("R")


def run_tasks(items: list[T], fn: Callable[[T], R], workers: int = 1) -> list[R]:
    """Run fn(item) for each item, returning results in the same order as items.

    workers <= 1 runs serially (safest for a single local llama.cpp server, and
    deterministic for tests). workers > 1 uses a small bounded thread pool — fine for
    cloud backends, whose calls are independent network requests. On the first
    exception, outstanding work is cancelled and the exception propagates immediately;
    nothing is swallowed.
    """
    if workers <= 1:
        return [fn(item) for item in items]

    results: list[R] = [None] * len(items)  # type: ignore[list-item]
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(fn, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()  # re-raises the worker's exception here
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results
