# paperdigest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLI that turns an arXiv paper into a folder of plain-English, wikilinked Obsidian markdown notes (concept chunks, ASCII diagrams, explained equations, global jargon glossary).

**Architecture:** Linear pipeline of plain functions over dataclasses: `fetch → extract → digest → render`. Two pluggable LLM backends (Anthropic; OpenAI-compatible which also covers local Ollama/vLLM via `base_url`). All tests run offline against fixtures and a fake backend.

**Tech Stack:** Python ≥3.11, typer, httpx, beautifulsoup4, anthropic, openai, pytest, Docker (python:3.12-slim).

## Global Constraints

- Project root: `~/code/paperdigest` (git repo already initialized; spec committed at `docs/superpowers/specs/2026-07-10-paperdigest-design.md`).
- Python ≥ 3.11 (uses `tomllib`, `X | None` unions).
- Runtime dependencies limited to: `typer`, `httpx`, `beautifulsoup4`, `anthropic`, `openai`. Dev dependency: `pytest`.
- API keys come ONLY from env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — never from config files or flags.
- No network and no API keys required by the test suite.
- Levels are exactly: `beginner`, `intermediate`, `advanced`. Default `intermediate`.
- Backends are exactly: `anthropic` (default), `openai`.
- Existing paper folders are never overwritten without `--force`; existing glossary term notes are NEVER overwritten, even with `--force`.
- All commands below run from `~/code/paperdigest`. Use a venv: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`; `pytest` below means `.venv/bin/pytest`.

---

### Task 1: Package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `paperdigest/__init__.py`
- Create: `.gitignore`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable package `paperdigest` with console script `paperdigest = paperdigest.cli:app` (cli module arrives in Task 8; script entry is declared now so install-time wiring is done once).

- [ ] **Step 1: Write the failing test**

`tests/test_scaffold.py`:

```python
def test_package_importable():
    import paperdigest

    assert paperdigest.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scaffold.py -v` (system python; venv not created yet)
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest'`

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:

```toml
[project]
name = "paperdigest"
version = "0.1.0"
description = "Turn AI/ML papers into plain-English Obsidian notes"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.12",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "anthropic>=0.40",
    "openai>=1.50",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
paperdigest = "paperdigest.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["paperdigest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`paperdigest/__init__.py`:

```python
__version__ = "0.1.0"
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
config.toml
```

- [ ] **Step 4: Create venv, install, run test to verify it passes**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/test_scaffold.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml paperdigest/__init__.py .gitignore tests/test_scaffold.py
git commit -m "feat: package scaffold with pyproject and console script"
```

---

### Task 2: config.py

**Files:**
- Create: `paperdigest/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `LEVELS: tuple[str, str, str]` = `("beginner", "intermediate", "advanced")`
  - `@dataclass Config(backend: str, model: str, base_url: str | None, vault: Path, level: str, max_input_chars: int, cache_dir: Path)`
  - `load_config(path: Path | None = None, **overrides) -> Config` — TOML file then keyword overrides; `None`-valued overrides are ignored; unknown TOML keys and invalid `level`/`backend` raise `ValueError`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:

```python
from pathlib import Path

import pytest

from paperdigest.config import Config, load_config


def test_defaults():
    cfg = load_config(None)
    assert cfg.backend == "anthropic"
    assert cfg.level == "intermediate"
    assert cfg.vault == Path("/vault")
    assert cfg.base_url is None


def test_toml_file_overrides_defaults(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('backend = "openai"\nvault = "/data/vault"\nlevel = "beginner"\n')
    cfg = load_config(f)
    assert cfg.backend == "openai"
    assert cfg.vault == Path("/data/vault")
    assert cfg.level == "beginner"


def test_kwargs_override_toml_and_none_is_ignored(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('level = "beginner"\n')
    cfg = load_config(f, level="advanced", backend=None)
    assert cfg.level == "advanced"
    assert cfg.backend == "anthropic"


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="level"):
        load_config(None, level="expert")


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        load_config(None, backend="gemini")


def test_unknown_toml_key_raises(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('api_key = "sk-nope"\n')
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.config'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/config.py`:

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

LEVELS = ("beginner", "intermediate", "advanced")
BACKENDS = ("anthropic", "openai")


@dataclass
class Config:
    backend: str = "anthropic"
    model: str = "claude-sonnet-5"
    base_url: str | None = None  # for local OpenAI-compatible servers (Ollama/vLLM)
    vault: Path = Path("/vault")
    level: str = "intermediate"
    max_input_chars: int = 400_000  # lower this for small-context local models
    cache_dir: Path = Path.home() / ".cache" / "paperdigest"


def load_config(path: Path | None = None, **overrides) -> Config:
    data: dict = {}
    if path is not None and path.exists():
        data = tomllib.loads(path.read_text())
    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    for key in ("vault", "cache_dir"):
        if key in merged:
            merged[key] = Path(merged[key])
    cfg = replace(Config(), **merged)
    if cfg.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {cfg.level!r}")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {cfg.backend!r}")
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/config.py tests/test_config.py
git commit -m "feat: config loading (TOML + overrides, validated)"
```

---

### Task 3: fetch.py

**Files:**
- Create: `paperdigest/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class FetchError(Exception)`
  - `parse_arxiv_id(ref: str) -> str` — accepts bare IDs (`1706.03762`, `1706.03762v7`) and arxiv.org abs/pdf/html URLs; raises `FetchError` otherwise.
  - `fetch_html(arxiv_id: str, cache_dir: Path, client: httpx.Client | None = None) -> str` — tries `arxiv.org/html/<id>` then `ar5iv.labs.arxiv.org/html/<id>`; validates the response is LaTeXML HTML (contains `ltx_title`); caches to `<cache_dir>/<id>.html`; raises `FetchError` with all attempted URLs on failure.

- [ ] **Step 1: Write the failing test**

`tests/test_fetch.py`:

```python
import httpx
import pytest

from paperdigest.fetch import FetchError, fetch_html, parse_arxiv_id


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/html/2405.12345v1", "2405.12345v1"),
    ],
)
def test_parse_arxiv_id(ref, expected):
    assert parse_arxiv_id(ref) == expected


def test_parse_rejects_non_arxiv():
    with pytest.raises(FetchError):
        parse_arxiv_id("https://example.com/1234.5678.pdf")
    with pytest.raises(FetchError):
        parse_arxiv_id("not a paper")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_uses_arxiv_html(tmp_path):
    def handler(request):
        assert "arxiv.org/html/1706.03762" in str(request.url)
        return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')

    html = fetch_html("1706.03762", tmp_path, client=_client(handler))
    assert "ltx_title" in html
    assert (tmp_path / "1706.03762.html").read_text() == html


def test_fetch_falls_back_to_ar5iv(tmp_path):
    def handler(request):
        if "ar5iv" in str(request.url):
            return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')
        return httpx.Response(404)

    html = fetch_html("1706.03762", tmp_path, client=_client(handler))
    assert "ltx_title" in html


def test_fetch_rejects_non_latexml_page(tmp_path):
    def handler(request):
        return httpx.Response(200, text="<html>abstract landing page</html>")

    with pytest.raises(FetchError, match="No HTML rendering"):
        fetch_html("1706.03762", tmp_path, client=_client(handler))


def test_fetch_reads_cache_without_network(tmp_path):
    (tmp_path / "1706.03762.html").write_text("cached!")

    def handler(request):
        raise AssertionError("network should not be hit")

    assert fetch_html("1706.03762", tmp_path, client=_client(handler)) == "cached!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.fetch'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/fetch.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

import httpx


class FetchError(Exception):
    pass


_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

_SOURCES = (
    "https://arxiv.org/html/{id}",
    "https://ar5iv.labs.arxiv.org/html/{id}",
)


def parse_arxiv_id(ref: str) -> str:
    ref = ref.strip()
    if "/" in ref and "arxiv.org" not in ref.lower():
        raise FetchError(f"not an arXiv reference: {ref!r} (v1 accepts arXiv URLs or IDs only)")
    m = _ID_RE.search(ref)
    if not m:
        raise FetchError(f"not an arXiv reference: {ref!r} (v1 accepts arXiv URLs or IDs only)")
    return m.group(1) + (m.group(2) or "")


def fetch_html(arxiv_id: str, cache_dir: Path, client: httpx.Client | None = None) -> str:
    cache_file = cache_dir / f"{arxiv_id}.html"
    if cache_file.exists():
        return cache_file.read_text()
    client = client or httpx.Client(follow_redirects=True, timeout=30)
    errors = []
    for tpl in _SOURCES:
        url = tpl.format(id=arxiv_id)
        try:
            resp = client.get(url)
        except httpx.HTTPError as e:
            errors.append(f"{url} -> {e}")
            continue
        if resp.status_code == 200 and "ltx_title" in resp.text:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text)
            return resp.text
        errors.append(f"{url} -> HTTP {resp.status_code}")
    raise FetchError(
        "No HTML rendering available for this paper (v1 supports arXiv HTML only).\n"
        "Tried:\n  " + "\n  ".join(errors)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_fetch.py -v`
Expected: 10 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/fetch.py tests/test_fetch.py
git commit -m "feat: arXiv HTML fetching with ar5iv fallback and disk cache"
```

---

### Task 4: extract.py + HTML fixture

**Files:**
- Create: `paperdigest/extract.py`
- Create: `tests/fixtures/paper.html`
- Create: `tests/conftest.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass Section(title: str, text: str)`
  - `@dataclass Paper(arxiv_id: str, title: str, abstract: str, sections: list[Section], url: str)`
  - `class ExtractError(Exception)`
  - `extract_paper(html: str, arxiv_id: str) -> Paper` — parses arXiv/ar5iv LaTeXML HTML; converts `<math alttext="...">` to `$...$`; drops the references/bibliography section; raises `ExtractError` if no title or no sections.
  - Test fixture `fixture_html` (conftest) returning the fixture file contents.

- [ ] **Step 1: Create the fixture and conftest**

`tests/fixtures/paper.html` (minimal LaTeXML shape matching real arXiv HTML output):

```html
<!DOCTYPE html>
<html>
<body>
<h1 class="ltx_title ltx_title_document">Tiny Transformers Explained</h1>
<div class="ltx_abstract">
  <h6 class="ltx_title">Abstract</h6>
  <p class="ltx_p">We study tiny transformers and show attention
  <math alttext="A=QK^{T}"><semantics><mrow><mi>A</mi></mrow></semantics></math>
  suffices for toy tasks.</p>
</div>
<section class="ltx_section" id="S1">
  <h2 class="ltx_title ltx_title_section">1 Introduction</h2>
  <p class="ltx_p">Transformers are neural networks. We define attention as
  <math alttext="\mathrm{softmax}(QK^{T})V"><semantics><mrow><mi>s</mi></mrow></semantics></math>
  throughout this paper.</p>
</section>
<section class="ltx_section" id="S2">
  <h2 class="ltx_title ltx_title_section">2 Method</h2>
  <p class="ltx_p">Our method stacks two attention layers with residual connections.</p>
</section>
<section class="ltx_section" id="bib">
  <h2 class="ltx_title">References</h2>
  <p class="ltx_p">[1] Someone et al. 2017.</p>
</section>
</body>
</html>
```

`tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Write the failing test**

`tests/test_extract.py`:

```python
import pytest

from paperdigest.extract import ExtractError, extract_paper


def test_extracts_title_abstract_sections(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert paper.title == "Tiny Transformers Explained"
    assert paper.abstract.startswith("We study tiny transformers")
    assert paper.url == "https://arxiv.org/abs/1706.03762"
    assert [s.title for s in paper.sections] == ["1 Introduction", "2 Method"]


def test_math_becomes_latex(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert "$A=QK^{T}$" in paper.abstract
    assert r"$\mathrm{softmax}(QK^{T})V$" in paper.sections[0].text


def test_references_section_dropped(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert not any("References" in s.title for s in paper.sections)


def test_non_paper_html_raises():
    with pytest.raises(ExtractError):
        extract_paper("<html><body><p>hello</p></body></html>", "1234.56789")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.extract'`

- [ ] **Step 4: Write minimal implementation**

`paperdigest/extract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup


class ExtractError(Exception):
    pass


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    sections: list[Section]
    url: str


def _mathify(soup: BeautifulSoup) -> None:
    for math in soup.find_all("math"):
        latex = math.get("alttext", "")
        math.replace_with(f"${latex}$" if latex else "")


def extract_paper(html: str, arxiv_id: str) -> Paper:
    soup = BeautifulSoup(html, "html.parser")
    _mathify(soup)

    title_el = soup.find(class_="ltx_title_document")
    if title_el is None:
        raise ExtractError("could not find a paper title — is this arXiv LaTeXML HTML?")
    title = title_el.get_text(" ", strip=True)

    abstract = ""
    abstract_el = soup.find(class_="ltx_abstract")
    if abstract_el is not None:
        heading = abstract_el.find(["h6", "h2", "h3"])
        if heading:
            heading.extract()
        abstract = abstract_el.get_text(" ", strip=True)

    sections: list[Section] = []
    for sec in soup.find_all("section", class_="ltx_section"):
        heading = sec.find(["h2", "h3"])
        sec_title = heading.get_text(" ", strip=True) if heading else "Untitled"
        if heading:
            heading.extract()
        low = sec_title.lower()
        if "bib" in (sec.get("id") or "") or low.startswith(("references", "bibliography", "acknowledg")):
            continue
        text = sec.get_text(" ", strip=True)
        if text:
            sections.append(Section(title=sec_title, text=text))
    if not sections:
        raise ExtractError("no sections found — is this arXiv LaTeXML HTML?")

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        sections=sections,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_extract.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
git add paperdigest/extract.py tests/fixtures/paper.html tests/conftest.py tests/test_extract.py
git commit -m "feat: LaTeXML HTML extraction to Paper dataclass"
```

---

### Task 5: llm.py backends

**Files:**
- Create: `paperdigest/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class LLMError(Exception)`
  - `class Backend(Protocol)` with `model: str` and `complete(self, system: str, user: str) -> str`
  - `AnthropicBackend(model: str)` — requires `ANTHROPIC_API_KEY` env; raises `LLMError` if missing.
  - `OpenAICompatibleBackend(model: str, base_url: str | None = None)` — requires `OPENAI_API_KEY` unless `base_url` is set (local servers don't need keys).
  - `make_backend(backend: str, model: str, base_url: str | None = None) -> Backend`
  - `complete_with_retry(backend: Backend, system: str, user: str, retries: int = 2) -> str` — exponential backoff, wraps final failure in `LLMError`.

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.llm'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/llm.py`:

```python
from __future__ import annotations

import os
import time
from typing import Protocol


class LLMError(Exception):
    pass


class Backend(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicBackend:
    def __init__(self, model: str):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
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
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def make_backend(backend: str, model: str, base_url: str | None = None) -> Backend:
    if backend == "anthropic":
        return AnthropicBackend(model)
    if backend == "openai":
        return OpenAICompatibleBackend(model, base_url)
    raise LLMError(f"unknown backend: {backend!r}")


def complete_with_retry(backend: Backend, system: str, user: str, retries: int = 2) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return backend.complete(system, user)
        except Exception as e:  # noqa: BLE001 — SDK exception types vary per backend
            last_error = e
            if attempt < retries:
                time.sleep(2**attempt)
    raise LLMError(f"LLM call failed after {retries + 1} attempts: {last_error}") from last_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_llm.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/llm.py tests/test_llm.py
git commit -m "feat: pluggable LLM backends (Anthropic + OpenAI-compatible) with retry"
```

---

### Task 6: digest.py

**Files:**
- Create: `paperdigest/digest.py`
- Test: `tests/test_digest.py`

**Interfaces:**
- Consumes: `Paper`, `Section` from `paperdigest.extract`; `Backend`, `LLMError`, `complete_with_retry` from `paperdigest.llm`; `LEVELS` semantics from config.
- Produces:
  - `@dataclass ConceptNote(title: str, body_md: str, section: str)`
  - `@dataclass Digest(paper: Paper, tldr: str, why_it_matters: str, concepts: list[ConceptNote], jargon: list[str], glossary: dict[str, str], self_test: list[str], model: str, level: str)` — `jargon` is ALL terms (for linking); `glossary` holds definitions for NEW terms only.
  - `build_digest(paper: Paper, backend: Backend, level: str, existing_terms: set[str], max_chars: int, progress: Callable[[str], None] = ...) -> Digest` — `existing_terms` are lowercase glossary-note stems already in the vault; if every jargon term already exists, the glossary LLM call is skipped entirely.

- [ ] **Step 1: Write the failing test**

`tests/test_digest.py`:

```python
import json

import pytest

from paperdigest.digest import Digest, build_digest, _call_json
from paperdigest.extract import Paper, Section
from paperdigest.llm import LLMError
from conftest import FakeBackend  # tests/ is on sys.path under pytest


@pytest.fixture
def paper():
    return Paper(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        abstract="We study tiny transformers.",
        sections=[
            Section("1 Introduction", "Transformers are neural networks."),
            Section("2 Method", "Our method stacks two attention layers."),
        ],
        url="https://arxiv.org/abs/1706.03762",
    )


OUTLINE = json.dumps(
    {
        "tldr": "Tiny transformers work.",
        "why_it_matters": "Cheap models are useful.",
        "concepts": [
            {"title": "Self-Attention", "section": "1 Introduction"},
            {"title": "Layer Stacking", "section": "2 Method"},
        ],
        "jargon": ["attention", "softmax"],
        "self_test": ["What is attention?", "Why stack layers?", "What is softmax?"],
    }
)

GLOSSARY = json.dumps({"terms": {"attention": "A weighting scheme.", "softmax": "Turns scores into probabilities."}})


def test_build_digest_happy_path(paper):
    backend = FakeBackend([OUTLINE, "Body about attention.", "Body about stacking.", GLOSSARY])
    d = build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None)
    assert isinstance(d, Digest)
    assert d.tldr == "Tiny transformers work."
    assert [c.title for c in d.concepts] == ["Self-Attention", "Layer Stacking"]
    assert d.concepts[0].body_md == "Body about attention."
    assert d.jargon == ["attention", "softmax"]
    assert d.glossary["softmax"] == "Turns scores into probabilities."
    assert len(d.self_test) == 3
    assert d.model == "fake-model"
    assert d.level == "intermediate"
    assert len(backend.calls) == 4  # outline + 2 concepts + glossary


def test_concept_call_gets_matching_section_text(paper):
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    build_digest(paper, backend, "beginner", set(), 400_000, progress=lambda m: None)
    concept1_user = backend.calls[1][1]
    assert "Transformers are neural networks." in concept1_user
    concept2_user = backend.calls[2][1]
    assert "stacks two attention layers" in concept2_user


def test_existing_terms_skip_glossary_call(paper):
    backend = FakeBackend([OUTLINE, "c1", "c2"])
    d = build_digest(paper, backend, "intermediate", {"attention", "softmax"}, 400_000, progress=lambda m: None)
    assert d.glossary == {}
    assert d.jargon == ["attention", "softmax"]  # still linkable
    assert len(backend.calls) == 3  # no glossary call


def test_long_paper_is_trimmed(paper):
    paper.sections[0].text = "x" * 100_000
    paper.sections[1].text = "y" * 100_000
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    warnings = []
    build_digest(paper, backend, "intermediate", set(), 10_000, progress=warnings.append)
    outline_user = backend.calls[0][1]
    assert len(outline_user) < 20_000
    assert any("trimm" in w.lower() for w in warnings)


def test_call_json_repairs_broken_json():
    backend = FakeBackend(['{"a": 1,,,}', '{"a": 1}'])
    assert _call_json(backend, "sys", "user") == {"a": 1}
    assert len(backend.calls) == 2


def test_call_json_strips_markdown_fences():
    backend = FakeBackend(['```json\n{"a": 1}\n```'])
    assert _call_json(backend, "sys", "user") == {"a": 1}


def test_call_json_raises_after_failed_repair():
    backend = FakeBackend(["nope", "still nope"])
    with pytest.raises(LLMError, match="unparseable JSON"):
        _call_json(backend, "sys", "user")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_digest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.digest'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/digest.py`:

```python
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Callable

from .extract import Paper
from .llm import Backend, LLMError, complete_with_retry


@dataclass
class ConceptNote:
    title: str
    body_md: str
    section: str


@dataclass
class Digest:
    paper: Paper
    tldr: str
    why_it_matters: str
    concepts: list[ConceptNote]
    jargon: list[str]           # every term of art in the paper (for linking)
    glossary: dict[str, str]    # definitions for terms NOT already in the vault
    self_test: list[str]
    model: str
    level: str


LEVEL_GUIDANCE = {
    "beginner": "Assume no ML background at all. Use everyday analogies. No unexplained jargon.",
    "intermediate": (
        "Assume a technically capable reader (writes code, knows basic statistics) who is not "
        "fluent in ML paper vocabulary. Plain English; define terms of art on first use."
    ),
    "advanced": (
        "Assume an ML practitioner. Be precise and compact; standard terminology is fine, "
        "but still explain the paper's novel jargon."
    ),
}

_OUTLINE_SYSTEM = """\
You are an expert teacher who breaks AI/ML research papers into teachable chunks for self-learners.
Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"tldr": "<2-3 plain-English sentences>",
 "why_it_matters": "<2-3 sentences on why this paper matters>",
 "concepts": [{"title": "<concept name>", "section": "<paper section heading it mainly comes from>"}],
 "jargon": ["<term>", ...],
 "self_test": ["<question>", ...]}
Rules: pick 4-8 concepts that each deserve their own explainer note.
"jargon" lists every term of art a newcomer would need defined (10-25 terms, lowercase).
"self_test" is 3-5 questions that check understanding of the whole paper."""

_CONCEPT_SYSTEM = """\
You write ONE markdown explainer note about a single concept from a research paper, for a self-learner.
Reader level: {guidance}
Rules:
- Plain English. Strip jargon or define it inline in parentheses.
- Use an analogy if it genuinely helps.
- If the concept is structural (architecture, data flow, training loop, algorithm), include an ASCII diagram in a fenced code block.
- If the concept involves key equations, show them in LaTeX ($...$ or $$...$$) followed by a line-by-line plain-English walkthrough.
- Cite where in the paper the idea comes from (e.g. "from §3.2").
Output ONLY the markdown body: no top-level heading, no frontmatter."""

_GLOSSARY_SYSTEM = """\
You define research-paper jargon in plain English for a {level}-level reader.
Respond with ONLY valid JSON (no markdown fences): {{"terms": {{"<term>": "<1-2 sentence plain-English definition>"}}}}.
Every requested term must appear as a key."""

_REPAIR_SYSTEM = "The user message was supposed to be valid JSON but is not. Return ONLY the corrected JSON, nothing else."


def _default_progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text


def _call_json(backend: Backend, system: str, user: str) -> dict:
    raw = complete_with_retry(backend, system, user)
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        repaired = complete_with_retry(backend, _REPAIR_SYSTEM, raw)
        try:
            return json.loads(_strip_fences(repaired))
        except json.JSONDecodeError as e:
            raise LLMError(f"model returned unparseable JSON even after a repair attempt: {e}\n---\n{raw[:2000]}") from e


def _paper_body(paper: Paper, max_chars: int, progress: Callable[[str], None]) -> str:
    parts = [f"## {s.title}\n{s.text}" for s in paper.sections]
    body = "\n\n".join(parts)
    if len(body) <= max_chars:
        return body
    progress(f"Warning: paper is {len(body)} chars; trimming sections to fit {max_chars}-char budget.")
    ratio = max_chars / len(body)
    trimmed = [f"## {s.title}\n{s.text[: max(200, int(len(s.text) * ratio))]}" for s in paper.sections]
    return "\n\n".join(trimmed)


def _find_section_text(paper: Paper, section_title: str, fallback: str) -> str:
    want = section_title.strip().lower()
    for s in paper.sections:
        have = s.title.strip().lower()
        if want and (want in have or have in want):
            return s.text
    return fallback


def build_digest(
    paper: Paper,
    backend: Backend,
    level: str,
    existing_terms: set[str],
    max_chars: int,
    progress: Callable[[str], None] = _default_progress,
) -> Digest:
    guidance = LEVEL_GUIDANCE[level]
    body = _paper_body(paper, max_chars, progress)

    progress(f"Outlining '{paper.title}'...")
    outline = _call_json(
        backend,
        _OUTLINE_SYSTEM,
        f"READER LEVEL: {level} — {guidance}\n\nPAPER TITLE: {paper.title}\n\n"
        f"ABSTRACT: {paper.abstract}\n\nFULL TEXT:\n{body}",
    )
    for key in ("tldr", "why_it_matters", "concepts", "jargon", "self_test"):
        if key not in outline:
            raise LLMError(f"outline response is missing the {key!r} field")

    concepts: list[ConceptNote] = []
    total = len(outline["concepts"])
    for i, c in enumerate(outline["concepts"], 1):
        title = c["title"]
        section = c.get("section", "")
        progress(f"Explaining concept {i}/{total}: {title}")
        section_text = _find_section_text(paper, section, fallback=body[:20_000])
        md = complete_with_retry(
            backend,
            _CONCEPT_SYSTEM.format(guidance=guidance),
            f"PAPER: {paper.title}\nPAPER TLDR: {outline['tldr']}\n"
            f"CONCEPT TO EXPLAIN: {title}\nSOURCE SECTION: {section}\n\n"
            f"ABSTRACT: {paper.abstract}\n\nSECTION TEXT:\n{section_text}",
        )
        concepts.append(ConceptNote(title=title, body_md=md.strip(), section=section))

    jargon = [str(t) for t in outline["jargon"]]
    new_terms = [t for t in jargon if t.lower() not in existing_terms]
    glossary: dict[str, str] = {}
    if new_terms:
        progress(f"Defining {len(new_terms)} glossary terms...")
        g = _call_json(
            backend,
            _GLOSSARY_SYSTEM.format(level=level),
            f"PAPER: {paper.title}\nABSTRACT: {paper.abstract}\n\nTERMS TO DEFINE:\n"
            + "\n".join(f"- {t}" for t in new_terms),
        )
        glossary = {str(k): str(v) for k, v in g.get("terms", {}).items()}

    return Digest(
        paper=paper,
        tldr=str(outline["tldr"]),
        why_it_matters=str(outline["why_it_matters"]),
        concepts=concepts,
        jargon=jargon,
        glossary=glossary,
        self_test=[str(q) for q in outline["self_test"]],
        model=backend.model,
        level=level,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_digest.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/digest.py tests/test_digest.py
git commit -m "feat: LLM digest orchestration (outline, concepts, glossary) with JSON repair"
```

---

### Task 7: render.py

**Files:**
- Create: `paperdigest/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Digest`, `ConceptNote` from `paperdigest.digest`; `Paper`, `Section` from `paperdigest.extract`.
- Produces:
  - `class OutputExistsError(Exception)`
  - `slugify(text: str) -> str` — lowercase, hyphenated, first 6 words.
  - `note_name(text: str) -> str` — filename-safe, keeps case/spaces, strips `\/:*?"<>|[]#^`.
  - `render_digest(digest: Digest, vault: Path, force: bool = False) -> Path` — writes `Papers/<year>-<slug>/00 Overview.md` + `NN <Concept>.md`, and `Glossary/<term>.md` for each NEW term; refuses existing paper folder unless `force`; NEVER overwrites existing glossary notes; returns the paper folder path. Year = `"20" + arxiv_id[:2]`.

- [ ] **Step 1: Write the failing test**

`tests/test_render.py`:

```python
import pytest

from paperdigest.digest import ConceptNote, Digest
from paperdigest.extract import Paper, Section
from paperdigest.render import OutputExistsError, note_name, render_digest, slugify


def _digest():
    paper = Paper(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        abstract="We study tiny transformers.",
        sections=[Section("1 Introduction", "text")],
        url="https://arxiv.org/abs/1706.03762",
    )
    return Digest(
        paper=paper,
        tldr="Tiny transformers work.",
        why_it_matters="Cheap models are useful.",
        concepts=[
            ConceptNote("Self-Attention", "Attention body.", "1 Introduction"),
            ConceptNote("Layer Stacking", "Stacking body.", "2 Method"),
        ],
        jargon=["attention", "softmax"],
        glossary={"attention": "A weighting scheme."},  # softmax pre-exists in vault
        self_test=["What is attention?"],
        model="fake-model",
        level="intermediate",
    )


def test_slugify_and_note_name():
    assert slugify("Attention Is All You Need, Really!") == "attention-is-all-you-need-really"
    assert note_name('Bad: "name" [x] #tag?') == "Bad name x tag"


def test_render_creates_folder_and_notes(tmp_path):
    folder = render_digest(_digest(), tmp_path)
    assert folder == tmp_path / "Papers" / "2017-tiny-transformers-explained"
    overview = (folder / "00 Overview.md").read_text()
    assert "Tiny transformers work." in overview
    assert "[[01 Self-Attention]]" in overview
    assert "[[02 Layer Stacking]]" in overview
    assert "[[attention]]" in overview and "[[softmax]]" in overview
    assert "What is attention?" in overview
    assert "source: https://arxiv.org/abs/1706.03762" in overview
    assert "level: intermediate" in overview
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "Attention body." in concept
    assert "1 Introduction" in concept


def test_render_writes_new_glossary_terms_only(tmp_path):
    gdir = tmp_path / "Glossary"
    gdir.mkdir(parents=True)
    (gdir / "softmax.md").write_text("MY HAND-WRITTEN NOTE")
    render_digest(_digest(), tmp_path)
    assert "A weighting scheme." in (gdir / "attention.md").read_text()
    assert (gdir / "softmax.md").read_text() == "MY HAND-WRITTEN NOTE"


def test_render_refuses_existing_folder(tmp_path):
    render_digest(_digest(), tmp_path)
    with pytest.raises(OutputExistsError, match="--force"):
        render_digest(_digest(), tmp_path)


def test_render_force_overwrites(tmp_path):
    folder = render_digest(_digest(), tmp_path)
    (folder / "00 Overview.md").write_text("my edits")
    folder2 = render_digest(_digest(), tmp_path, force=True)
    assert folder2 == folder
    assert "Tiny transformers work." in (folder / "00 Overview.md").read_text()


def test_force_still_never_touches_glossary(tmp_path):
    render_digest(_digest(), tmp_path)
    gnote = tmp_path / "Glossary" / "attention.md"
    gnote.write_text("MY EDITS")
    render_digest(_digest(), tmp_path, force=True)
    assert gnote.read_text() == "MY EDITS"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.render'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/render.py`:

```python
from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path

from .digest import Digest


class OutputExistsError(Exception):
    pass


def slugify(text: str, max_words: int = 6) -> str:
    words = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "untitled"


def note_name(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\[\]#^]', "", text)
    return re.sub(r"\s+", " ", cleaned).strip() or "Untitled"


def _frontmatter(digest: Digest, extra_tags: str = "") -> str:
    today = datetime.date.today().isoformat()
    tags = f"[paper-digest{extra_tags}]"
    return (
        "---\n"
        f"source: {digest.paper.url}\n"
        f"date: {today}\n"
        f"model: {digest.model}\n"
        f"level: {digest.level}\n"
        f"tags: {tags}\n"
        "---\n\n"
    )


def render_digest(digest: Digest, vault: Path, force: bool = False) -> Path:
    year = "20" + digest.paper.arxiv_id[:2]
    folder = vault / "Papers" / f"{year}-{slugify(digest.paper.title)}"
    if folder.exists():
        if not force:
            raise OutputExistsError(
                f"{folder} already exists — re-run with --force to overwrite (your edits there will be lost)"
            )
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    concept_names = [f"{i:02d} {note_name(c.title)}" for i, c in enumerate(digest.concepts, 1)]

    fm = _frontmatter(digest)
    overview = fm + f"# {digest.paper.title}\n\n"
    overview += f"**TLDR:** {digest.tldr}\n\n"
    overview += f"## Why it matters\n\n{digest.why_it_matters}\n\n"
    overview += "## Concepts\n\n" + "\n".join(f"- [[{n}]]" for n in concept_names) + "\n\n"
    if digest.jargon:
        links = " · ".join(f"[[{note_name(t.lower())}]]" for t in digest.jargon)
        overview += f"## Glossary terms\n\n{links}\n\n"
    if digest.self_test:
        overview += "## Check your understanding\n\n"
        overview += "\n".join(f"{i}. {q}" for i, q in enumerate(digest.self_test, 1)) + "\n"
    (folder / "00 Overview.md").write_text(overview)

    for name, concept in zip(concept_names, digest.concepts):
        note = fm + f"# {note_name(concept.title)}\n\n"
        if concept.section:
            note += f"*Source: {concept.section} of [{digest.paper.title}]({digest.paper.url})*\n\n"
        note += concept.body_md + "\n"
        (folder / f"{name}.md").write_text(note)

    if digest.glossary:
        gdir = vault / "Glossary"
        gdir.mkdir(parents=True, exist_ok=True)
        gfm = _frontmatter(digest, extra_tags=", glossary")
        for term, definition in digest.glossary.items():
            gfile = gdir / f"{note_name(term.lower())}.md"
            if gfile.exists():
                continue  # never clobber accumulated/edited term notes
            gfile.write_text(gfm + f"# {note_name(term)}\n\n{definition}\n")

    return folder
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_render.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/render.py tests/test_render.py
git commit -m "feat: markdown rendering with wikilinks, force guard, global glossary"
```

---

### Task 8: cli.py + offline end-to-end test

**Files:**
- Create: `paperdigest/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above. Imports: `from .config import load_config`; `from .llm import LLMError, make_backend`; `from . import digest as digest_mod`; `from . import extract, fetch, render` (module imports so tests can monkeypatch `cli.fetch.fetch_html` and `cli.make_backend`).
- Produces: `app` (typer.Typer) — console script entry. Usage: `paperdigest <ref> [--level L] [--backend B] [--model M] [--base-url U] [--vault P] [--config P] [--force]`. Prints the created folder path to stdout; progress to stderr; known errors print `error: ...` to stderr and exit 1.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:

```python
import json

from typer.testing import CliRunner

from paperdigest import cli
from conftest import FakeBackend  # tests/ is on sys.path under pytest

runner = CliRunner()

OUTLINE = json.dumps(
    {
        "tldr": "Tiny transformers work.",
        "why_it_matters": "Cheap models are useful.",
        "concepts": [{"title": "Self-Attention", "section": "1 Introduction"}],
        "jargon": ["attention"],
        "self_test": ["What is attention?"],
    }
)
GLOSSARY = json.dumps({"terms": {"attention": "A weighting scheme."}})


def _fake_responses():
    return [OUTLINE, "Attention body.", GLOSSARY]


def _patch(monkeypatch, fixture_html):
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend(_fake_responses()))


def test_end_to_end_offline(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "Papers" / "2017-tiny-transformers-explained"
    assert (folder / "00 Overview.md").exists()
    assert (folder / "01 Self-Attention.md").exists()
    assert (tmp_path / "Glossary" / "attention.md").exists()


def test_rerun_refuses_without_force(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    assert runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_rerun_with_force_succeeds(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    assert runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output


def test_existing_glossary_terms_skip_llm_definitions(tmp_path, monkeypatch, fixture_html):
    gdir = tmp_path / "Glossary"
    gdir.mkdir(parents=True)
    (gdir / "attention.md").write_text("MINE")
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend([OUTLINE, "Attention body."]))
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (gdir / "attention.md").read_text() == "MINE"


def test_bad_ref_errors_cleanly(tmp_path):
    result = runner.invoke(cli.app, ["https://example.com/nope", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert "error:" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.cli'`

- [ ] **Step 3: Write minimal implementation**

`paperdigest/cli.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import digest as digest_mod
from . import extract, fetch, render
from .config import load_config
from .llm import LLMError, make_backend

app = typer.Typer(add_completion=False)


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


@app.command()
def main(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    level: str = typer.Option(None, "--level", help="beginner | intermediate | advanced"),
    backend: str = typer.Option(None, "--backend", help="anthropic | openai (openai also covers local servers)"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible server URL, e.g. http://localhost:11434/v1"),
    vault: Path = typer.Option(None, "--vault", help="Obsidian vault root (notes go to Papers/ and Glossary/)"),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing paper folder"),
):
    """Digest an AI/ML paper into plain-English Obsidian notes."""
    try:
        cfg = load_config(
            config,
            level=level,
            backend=backend,
            model=model,
            base_url=base_url,
            vault=vault,
        )
        arxiv_id = fetch.parse_arxiv_id(ref)
        _progress(f"Fetching arXiv {arxiv_id}...")
        html = fetch.fetch_html(arxiv_id, cfg.cache_dir)
        paper = extract.extract_paper(html, arxiv_id)
        _progress(f"Extracted '{paper.title}' ({len(paper.sections)} sections)")
        llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url)
        gdir = cfg.vault / "Glossary"
        existing_terms = {p.stem.lower() for p in gdir.glob("*.md")} if gdir.exists() else set()
        d = digest_mod.build_digest(
            paper, llm_backend, cfg.level, existing_terms, cfg.max_input_chars, progress=_progress
        )
        folder = render.render_digest(d, cfg.vault, force=force)
        print(folder)
    except (fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS (scaffold, config, fetch, extract, llm, digest, render, cli)

- [ ] **Step 5: Commit**

```bash
git add paperdigest/cli.py tests/test_cli.py
git commit -m "feat: CLI wiring with offline end-to-end tests"
```

---

### Task 9: Dockerfile, config.example.toml, README

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `config.example.toml`
- Create: `README.md`

**Interfaces:**
- Consumes: the installed package + console script from Tasks 1–8.
- Produces: `paperdigest` Docker image with `ENTRYPOINT ["paperdigest"]` and `/vault` volume.

- [ ] **Step 1: Write the files**

`Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY paperdigest ./paperdigest
RUN pip install --no-cache-dir .

VOLUME /vault

ENTRYPOINT ["paperdigest"]
CMD ["--help"]
```

`.dockerignore`:

```
.venv
.git
tests
docs
__pycache__
*.egg-info
```

`config.example.toml`:

```toml
# Copy to config.toml and adjust. API keys come from env vars only:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY

backend = "anthropic"          # "anthropic" or "openai" (openai also covers local servers)
model = "claude-sonnet-5"
# base_url = "http://localhost:11434/v1"   # for local Ollama/vLLM, with backend = "openai"
vault = "/vault"               # your Obsidian vault root
level = "intermediate"         # beginner | intermediate | advanced
# max_input_chars = 24000      # lower for small-context local models
```

`README.md`:

````markdown
# paperdigest

Turn an AI/ML arXiv paper into a folder of plain-English, wikilinked Obsidian
notes: concept-by-concept explainers, ASCII diagrams, equations walked through
line by line, and a vault-wide jargon glossary.

## Quick start (local install)

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 --vault ~/ObsidianVault --level intermediate
```

Output lands in `<vault>/Papers/<year>-<title-slug>/` plus new term notes in
`<vault>/Glossary/`. Re-running the same paper refuses to overwrite unless you
pass `--force`. Glossary term notes are never overwritten.

## Docker

```bash
docker build -t paperdigest .
docker run --rm -v ~/ObsidianVault:/vault -e ANTHROPIC_API_KEY \
  paperdigest https://arxiv.org/abs/1706.03762 --level beginner
```

## Backends

| Backend | Flags | Key |
|---|---|---|
| Claude (default) | `--backend anthropic --model claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `--backend openai --model gpt-5` | `OPENAI_API_KEY` |
| Local (Ollama/vLLM) | `--backend openai --model llama3.1 --base-url http://localhost:11434/v1` | none |

For local models with small context windows, set `max_input_chars` (e.g.
24000) in `config.toml` — long papers get proportionally trimmed with a
warning. From Docker, reach a host-side local server with
`--base-url http://host.docker.internal:11434/v1` (add
`--add-host=host.docker.internal:host-gateway` on Linux).

## Config

Copy `config.example.toml` to `config.toml` (or pass `--config`). CLI flags
override config values. API keys are env-only.

## Inputs

v1 accepts arXiv URLs (`abs`/`pdf`/`html`) or bare IDs (`1706.03762`). It uses
the paper's arXiv HTML rendering (ar5iv as fallback); papers with no HTML
rendering fail with a clear message. Fetched papers are cached in
`~/.cache/paperdigest/`.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # fully offline: fixtures + fake LLM backend
```
````

- [ ] **Step 2: Verify Docker build and containerized help**

Run:
```bash
docker build -t paperdigest . && docker run --rm paperdigest
```
Expected: image builds; running it prints the typer help text (Usage: paperdigest ...).

If Docker is unavailable on the machine, note that in the commit message and verify instead with `.venv/bin/paperdigest --help`.

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/pytest`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore config.example.toml README.md
git commit -m "feat: Dockerfile, example config, README"
```

---

### Task 10: Live smoke test (manual, requires a real backend)

**Files:**
- Create: `docs/superpowers/smoke-test.md` (record of the run)

**Interfaces:**
- Consumes: the finished CLI.
- Produces: verified end-to-end behavior against a real paper and backend.

- [ ] **Step 1: Pick a backend available on this machine**

Check what's available (in this order):
```bash
env | grep -E "ANTHROPIC_API_KEY|OPENAI_API_KEY" | sed 's/=.*/=<set>/'
curl -s http://localhost:11434/v1/models 2>/dev/null | head -c 200   # Ollama?
```
If no key and no local server is available, STOP and ask the user which backend/key to use. Do not fabricate a smoke test result.

- [ ] **Step 2: Run against a real paper into a scratch vault**

```bash
mkdir -p /tmp/smoke-vault
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 --vault /tmp/smoke-vault --level intermediate
```
Expected: progress lines on stderr (Fetching → Extracted → Outlining → Explaining concept i/N → Defining terms), folder path on stdout.

- [ ] **Step 3: Inspect the artifacts**

```bash
find /tmp/smoke-vault -name "*.md" | sort
head -40 "/tmp/smoke-vault/Papers/"*/"00 Overview.md"
```
Verify by reading: overview has TLDR/concept wikilinks/self-test questions; at least one concept note has an ASCII diagram or explained equation; glossary notes exist.

- [ ] **Step 4: Verify re-run guard live**

```bash
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 --vault /tmp/smoke-vault; echo "exit=$?"
```
Expected: `error: ... --force ...` and `exit=1`.

- [ ] **Step 5: Record results and commit**

Write what was run, which backend/model, and what was verified (quote a few artifact lines) to `docs/superpowers/smoke-test.md`.

```bash
git add docs/superpowers/smoke-test.md
git commit -m "test: record live smoke test results"
```
